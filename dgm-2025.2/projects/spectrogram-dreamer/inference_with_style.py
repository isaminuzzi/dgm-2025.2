import torch
import torchaudio.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

# Fix for plotting on headless servers
matplotlib.use('Agg')

from src.models.dreamer import DreamerModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_RATE = 16000 
N_FFT = 512
WIN_LENGTH = 20
HOP_LENGTH = 10 
N_MELS = 64
F_MIN = 50
F_MAX = 7600

# ==========================================
# GLOBAL STATS
# ==========================================
GLOBAL_MEAN = np.array([-5.734388, -5.6716228, -4.7026305, -4.967197, -4.9417186, -4.808997, -5.3056636, -5.224505, -5.3299317, -5.4567504, -5.640661, -5.778809, -6.1192017, -6.314107, -6.6071506, -6.840761, -7.125506, -7.35195, -7.525212, -7.6915693, -7.758261, -7.9011807, -8.121692, -8.076018, -8.291807, -8.216392, -8.324706, -8.257913, -8.253128, -8.255108, -8.279936, -8.346785, -8.439514, -8.538402, -8.627999, -8.69899, -8.688386, -8.709745, -8.690824, -8.757223, -8.831239, -8.966529, -9.043103, -9.083665, -9.097104, -9.107875, -9.151038, -9.26776, -9.356491, -9.483289, -9.573892, -9.670915, -9.670915, -9.763971, -9.848085, -9.934736, -9.993131, -10.063034, -10.113865, -10.138746, -10.148016, -10.168127, -10.2440815, -10.356394, -10.500871], dtype=np.float32)
if len(GLOBAL_MEAN) != N_MELS: GLOBAL_MEAN = np.resize(GLOBAL_MEAN, N_MELS)
GLOBAL_STD = np.array([4.3592706, 4.3777413, 4.7913637, 4.7612023, 4.8177733, 4.933831, 4.7991247, 4.803811, 4.8006306, 4.809823, 4.800325, 4.7702727, 4.6295657, 4.5106544, 4.3712707, 4.2459526, 4.091292, 3.9534912, 3.8408408, 3.743386, 3.6955655, 3.6256783, 3.530331, 3.5458634, 3.4443364, 3.4620464, 3.4033048, 3.4320881, 3.4317906, 3.4260743, 3.4161582, 3.3840473, 3.332779, 3.2681808, 3.2089667, 3.1718438, 3.180097, 3.17876, 3.2006915, 3.168355, 3.1110375, 3.0211756, 2.9824722, 2.9754653, 2.975464, 2.9738736, 2.958073, 2.8942401, 2.842133, 2.7727149, 2.736065, 2.6963437, 2.655312, 2.6240668, 2.5850794, 2.5561945, 2.5039523, 2.4592898, 2.4284506, 2.4086397, 2.3757217, 2.3001747, 2.1992095, 2.0644271], dtype=np.float32)
if len(GLOBAL_STD) != N_MELS: GLOBAL_STD = np.resize(GLOBAL_STD, N_MELS)

# ==========================================
# 1. STYLE ENGINE
# ==========================================
class StyleEngine:
    def __init__(self, device):
        self.device = device

    def load_style_from_file(self, pt_path):
        try:
            vec = torch.load(pt_path, map_location=self.device)
            if vec.dim() == 2: vec = vec.squeeze(0)
            return vec.float()
        except Exception as e:
            logger.error(f"Could not load style {pt_path}: {e}")
            return torch.zeros(53, device=self.device)

    def create_hybrid_style(self, style_a_path, style_b_path, blend_ratio=0.5):
        vec_a = self.load_style_from_file(style_a_path)
        vec_b = self.load_style_from_file(style_b_path)
        hybrid = (vec_a * (1 - blend_ratio)) + (vec_b * blend_ratio)
        return hybrid

    def modify_prosody(self, base_vector, energy_scale=1.0, texture_scale=1.0):
        new_vec = base_vector.clone()
        new_vec[0] = new_vec[0] * energy_scale
        new_vec[1:21] = new_vec[1:21] * texture_scale
        return new_vec

# ==========================================
# 2. AUDIO PROCESSOR (With Quality Fixes)
# ==========================================
class AudioProcessor:
    def __init__(self, use_log=True, segment_duration=0.1, overlap=0.5):
        self.sr = SAMPLE_RATE
        self.n_fft = N_FFT
        self.n_mels = N_MELS
        self.f_min = F_MIN
        self.f_max = F_MAX
        self.use_log = use_log
        self.hop_len = int(self.sr * (HOP_LENGTH / 1000))
        self.win_len = int(self.sr * (WIN_LENGTH / 1000))
        self.frames_per_seg = int((segment_duration * self.sr) / self.hop_len)
        self.overlap = overlap

        if self.n_fft < self.win_len: self.n_fft = 1 << (self.win_len - 1).bit_length()

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sr, n_fft=self.n_fft, win_length=self.win_len,
            hop_length=self.hop_len, n_mels=self.n_mels, f_min=self.f_min, f_max=self.f_max,
            center=True, pad_mode="reflect", power=2.0, normalized=False, norm="slaney", mel_scale="htk"
        )
        self.inv_mel_transform = T.InverseMelScale(
            n_stft=self.n_fft // 2 + 1, n_mels=self.n_mels, sample_rate=self.sr,
            f_min=self.f_min, f_max=self.f_max, norm="slaney", mel_scale="htk", driver="gelss" 
        )
        self.griffin_lim = T.GriffinLim(
            n_fft=self.n_fft, win_length=self.win_len, hop_length=self.hop_len, power=1.0, n_iter=100 
        )

    def audio_to_segments(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sr: waveform = T.Resample(sr, self.sr)(waveform)
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)

        spec = self.mel_transform(waveform)
        if self.use_log: spec = torch.log(torch.clamp(spec, min=1e-5))
        
        C, H, Time = spec.shape
        W = self.frames_per_seg
        stride = int(W * (1 - self.overlap))
        segments = [spec[:, :, i : i + W] for i in range(0, Time - W + 1, stride)]
        
        if not segments: return torch.zeros(1, 1, C, H, W)
        return torch.stack(segments).unsqueeze(0)

    def segments_to_audio(self, segments_tensor, temperature=1.0, auto_scale=True):
        if segments_tensor.dim() == 5: segments_tensor = segments_tensor.squeeze(0)
        T_seq, C, H, W = segments_tensor.shape
        stride = int(W * (1 - self.overlap))
        
        total_len = (T_seq - 1) * stride + W
        full_spec = torch.zeros((C, H, total_len), device=segments_tensor.device)
        counts = torch.zeros((C, H, total_len), device=segments_tensor.device)
        
        # 1. Overlap Add
        for t in range(T_seq):
            start = t * stride
            full_spec[:, :, start:start+W] += segments_tensor[t]
            counts[:, :, start:start+W] += 1.0
            
        full_spec = full_spec / torch.clamp(counts, min=1.0)
        
        # 2. Temperature (CONTRAST KNOB)
        # This is where the magic happens. Multipling the Log-Spec expands/contracts dynamic range.
        if temperature != 1.0: 
            full_spec = full_spec * temperature

        # 3. Auto-Scaling
        if self.use_log and auto_scale:
            spec_flat = full_spec.reshape(-1)
            k = int(0.98 * spec_flat.numel())
            peak_val = torch.kthvalue(spec_flat, k).values if k < spec_flat.numel() else spec_flat.max()
            target_peak = -0.5
            if peak_val < target_peak:
                shift = torch.clamp(target_peak - peak_val, max=30.0) 
                full_spec = full_spec + shift

        if self.use_log: mel_power = torch.exp(full_spec)
        else: mel_power = full_spec
            
        # DSP Clean up
        mel_power = torch.where(mel_power < 2e-5, torch.zeros_like(mel_power), mel_power)
        mid_mel = self.n_mels // 2
        boost_curve = torch.linspace(1.0, 1.5, steps=self.n_mels - mid_mel, device=mel_power.device)
        mel_power[:, mid_mel:, :] = mel_power[:, mid_mel:, :] * boost_curve.unsqueeze(1)
        mel_power = torch.pow(mel_power, 1.3)

        try:
            linear_power = self.inv_mel_transform(mel_power)
        except RuntimeError:
            return torch.zeros(int(self.sr * 0.1))

        linear_power = torch.clamp(linear_power, min=1e-8)
        waveform = self.griffin_lim(torch.sqrt(linear_power))
        waveform = F.highpass_biquad(waveform, self.sr, cutoff_freq=60)
        peak = torch.abs(waveform).max()
        if peak > 0: waveform = waveform * (0.95 / peak)
        return waveform

# ==========================================
# 3. INFERENCE CLASS
# ==========================================
class SpectrogramDreamerInference:
    def __init__(self, model_path, use_log=True, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.processor = AudioProcessor(use_log=use_log) 
        self.use_log = use_log
        self.mean = torch.tensor(GLOBAL_MEAN, device=device).view(1, 1, 1, -1, 1)
        self.std = torch.tensor(GLOBAL_STD, device=device).view(1, 1, 1, -1, 1)
        
        self.model = DreamerModel(
            h_state_size=200, z_state_size=30, action_size=53, 
            embedding_size=256, in_channels=1, cnn_depth=32
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def denormalize(self, spec):
        return (spec * (self.std + 1e-6)) + self.mean

    def normalize(self, spec):
        return (spec - self.mean) / (self.std + 1e-6)

    def save_spectrogram_plot(self, tensor, filename, title="Spectrogram"):
        try:
            if tensor.dim() == 5: tensor = tensor.squeeze(0).squeeze(1)
            if tensor.dim() == 3: full_spec = torch.cat([t for t in tensor], dim=1)
            else: full_spec = tensor
            spec_np = full_spec.detach().cpu().numpy()
            if self.use_log: spec_db = 4.343 * spec_np
            else: spec_db = 10 * np.log10(np.maximum(spec_np, 1e-10))
            spec_db_norm = spec_db - np.max(spec_db)
            
            plt.figure(figsize=(12, 4))
            plt.imshow(spec_db_norm, aspect='auto', origin='lower', cmap='magma', 
                       interpolation='nearest', vmin=-80, vmax=0)
            plt.colorbar(format='%+2.0f dB')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
        except: pass

    def reconstruct(self, audio_path, out_dir, style_vector=None, temperature=1.0):
        logger.info(f"\n🏗️  RECONSTRUCTION (Temp: {temperature})")
        
        # 1. Load Input Audio
        raw = self.processor.audio_to_segments(audio_path).to(self.device)
        self.save_spectrogram_plot(raw, out_dir / "input.png", "Raw Input")
        
        norm = self.normalize(raw)
        B, T = norm.shape[:2]
        
        # 2. Prepare Actions
        if style_vector is not None:
            if style_vector.dim() == 1: style_vector = style_vector.unsqueeze(0)
            actions = style_vector.view(1, 1, -1).expand(B, T, -1).to(self.device)
        else:
            actions = torch.zeros(B, T, 53).to(self.device)

        # 3. Forward Pass
        with torch.no_grad():
            out = self.model(norm, actions, compute_loss=False)
            norm_recon = out['reconstructed']
            
        raw_recon = self.denormalize(norm_recon)
        self.save_spectrogram_plot(raw_recon, out_dir / "recon_styled.png", "Styled Reconstruction")
        
        # 4. Generate Audio with Temperature
        return self.processor.segments_to_audio(raw_recon, temperature, auto_scale=True)


# ==========================================
# 4. MAIN INTERFACE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Dreamer Audio Reconstruction with Prosody Control")
    parser.add_argument('--model', type=str, required=True, help='Path to .pt model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input audio file to reconstruct')
    parser.add_argument('--output', type=str, default='output', help='Output folder')
    
    # Style Modes
    parser.add_argument('--mode', type=str, default='transfer', choices=['transfer', 'blend', 'manual', 'neutral'], 
                        help='Style source: transfer, blend, manual, or neutral')

    # Style arguments
    parser.add_argument('--style_a', type=str, help='Path to .pt style vector')
    parser.add_argument('--style_b', type=str, help='Path to 2nd .pt style vector (for blend)')
    parser.add_argument('--blend_ratio', type=float, default=0.5, help='0.0 = Style A, 1.0 = Style B')
    
    # Prosody knobs
    parser.add_argument('--energy', type=float, default=1.0, help='Energy multiplier (Loudness)')
    parser.add_argument('--texture', type=float, default=1.0, help='Texture/Delta multiplier (Roughness)')
    
    # CRITICAL ADDITION
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Sampling temperature. >1.0 sharpens/increases contrast. <1.0 smoothes.')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    engine = StyleEngine(device)
    inference = SpectrogramDreamerInference(args.model, use_log=True, device=device)
    
    # --- CONSTRUCT THE STYLE VECTOR ---
    final_style = None
    
    if args.mode == 'neutral':
        final_style = torch.zeros(53).to(device)
    elif args.mode == 'transfer':
        if not args.style_a: raise ValueError("For transfer mode, provide --style_a")
        final_style = engine.load_style_from_file(args.style_a)
    elif args.mode == 'blend':
        if not args.style_a or not args.style_b: raise ValueError("For blend mode, provide --style_a and --style_b")
        final_style = engine.create_hybrid_style(args.style_a, args.style_b, args.blend_ratio)
    elif args.mode == 'manual':
        final_style = torch.zeros(53).to(device)

    # --- APPLY PROSODY MODIFIERS ---
    if final_style is not None:
        final_style = engine.modify_prosody(final_style, energy_scale=args.energy, texture_scale=args.texture)
    
    # --- RECONSTRUCT ---
    waveform = inference.reconstruct(
        audio_path=args.input,
        out_dir=out_dir,
        style_vector=final_style,
        temperature=args.temperature # Passed correctly from CLI
    )
    
    out_path = out_dir / f"recon_{args.mode}_temp{args.temperature}.wav"
    sf.write(out_path, waveform.squeeze().cpu().numpy(), SAMPLE_RATE)
    logger.info(f"✅ Reconstructed audio saved to {out_path}")

if __name__ == "__main__":
    main()