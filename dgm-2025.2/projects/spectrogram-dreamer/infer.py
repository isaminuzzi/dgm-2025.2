import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import tempfile
import os
import shutil

# Import your classes
from src.models.dreamer import DreamerModel 
from src.preprocessing.generate_spectrogram import AudioFile

def log_mel_to_waveform(spectrogram, n_fft, win_length, hop_length, sample_rate, n_mels, f_min, f_max):
    device = spectrogram.device
    
    # 1. Inverse Log: exp(x)
    power_spec = torch.exp(spectrogram)
    
    # 2. Inverse Mel Scale
    inverse_mel = T.InverseMelScale(
        sample_rate=sample_rate,
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        norm="slaney",
        mel_scale="htk",
        driver='gelsd' 
    ).to(device)
    
    try:
        linear_spec = inverse_mel(power_spec)
    except Exception as e:
        print(f"InverseMelScale Warning: {e}. Trying simplified inversion.")
        linear_spec = torch.matmul(inverse_mel.fb.pinverse(), power_spec)

    # 3. Griffin-Lim
    griffin_lim = T.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=2.0,
        n_iter=64 
    ).to(device)
    
    waveform = griffin_lim(linear_spec)
    return waveform

class StyleTransferInference:
    def __init__(self, model_path, action_size=53, n_mels=80, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.config = {
            'h_state_size': 200,
            'z_state_size': 30,
            'action_size': action_size,
            'embedding_size': 256,
            'aux_size': 5,
            'in_channels': 1,
            'cnn_depth': 32,
            'input_shape': (n_mels, 10)
        }
        
        self.model = DreamerModel(**self.config).to(self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def load_styles(self, styles_dir):
        path = Path(styles_dir)
        files = sorted(list(path.glob("*.pt")))
        if not files: raise ValueError(f"No .pt files found in {styles_dir}")
        vectors = [torch.load(f, map_location=self.device).squeeze() for f in files]
        return torch.stack(vectors)

    def preprocess_audio(self, audio_path, target_sr=22050):
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            print(f"Resampling input from {sr}Hz to {target_sr}Hz...")
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        torchaudio.save(temp_path, waveform, target_sr)
        return temp_path

    def run(self, audio_path, styles_path, output_path):
        out_path_obj = Path(output_path)
        base_name = out_path_obj.stem
        out_dir = out_path_obj.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        target_sr = 22050 
        temp_audio_path = self.preprocess_audio(audio_path, target_sr=target_sr)
        
        try:
            n_mels = self.config['input_shape'][0]
            
            # Setup AudioFile with 0 overlap for clean stitching
            audio = AudioFile(
                waveform_path=temp_audio_path,
                n_fft=512,        
                win_length=20,    
                hop_length=10, 
                n_mels=n_mels, 
                f_min=50, 
                f_max=7600,
                segment_duration=0.1, 
                overlap=0.0 
            )
            
            raw_segments = audio.segment_spectrogram()
            style_tensor = self.load_styles(styles_path)
            
            n_steps = min(len(raw_segments), len(style_tensor))
            print(f"Aligned to {n_steps} steps (approx {n_steps*0.1:.1f} seconds)")

            # Stitch original for visualization
            original_spectrogram_stitched = torch.cat(raw_segments[:n_steps], dim=1)

            obs_input = torch.stack(raw_segments[:n_steps]).to(self.device).unsqueeze(0).unsqueeze(2)
            act_input = style_tensor[:n_steps].to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(obs_input, act_input, compute_loss=False)
            
            reconstructed = outputs['reconstructed'].squeeze(0).squeeze(1)
            generated_spectrogram_stitched = torch.cat([seg for seg in reconstructed], dim=1)
            
            # --- FIXED SAVING BLOCK ---

            # A. Save Original Audio
            orig_audio_out = out_dir / f"{base_name}_original.wav"
            shutil.copy(temp_audio_path, str(orig_audio_out)) # Cast to str
            print(f"Saved Original Audio: {orig_audio_out}")

            # B. Save Generated Audio
            print("Converting generated spectrogram to waveform...")
            waveform = log_mel_to_waveform(
                generated_spectrogram_stitched,
                n_fft=audio.n_fft, win_length=audio.win_length, hop_length=audio.hop_length,
                sample_rate=audio.sample_rate, n_mels=n_mels, f_min=audio.f_min, f_max=audio.f_max
            )
            gen_audio_out = out_dir / f"{base_name}_generated.wav"
            
            # FIX: Cast Path to str() for torchaudio
            torchaudio.save(str(gen_audio_out), waveform.unsqueeze(0).cpu(), audio.sample_rate)
            print(f"Saved Generated Audio: {gen_audio_out}")

            # C. Save Comparison Plot
            print("Generating comparison plot...")
            fig, ax = plt.subplots(2, 1, figsize=(15, 10))
            
            orig_vis = original_spectrogram_stitched.cpu().numpy()
            ax[0].imshow(orig_vis, origin='lower', aspect='auto', cmap='magma')
            ax[0].set_title("Original Spectrogram (Input)")
            ax[0].axis('off')
            
            gen_vis = generated_spectrogram_stitched.cpu().numpy()
            ax[1].imshow(gen_vis, origin='lower', aspect='auto', cmap='magma')
            ax[1].set_title("Reconstructed/Style-Transferred Spectrogram (Output)")
            ax[1].axis('off')
            
            plot_out = out_dir / f"{base_name}_comparison.png"
            plt.tight_layout()
            plt.savefig(str(plot_out)) # Cast to str
            plt.close()
            print(f"Saved Comparison Plot: {plot_out}")
            
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--styles", required=True)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--n_mels", type=int, default=80) 
    parser.add_argument("--action_size", type=int, default=53)

    args = parser.parse_args()
    
    inferencer = StyleTransferInference(
        args.model, 
        action_size=args.action_size, 
        n_mels=args.n_mels
    )
    inferencer.run(args.audio, args.styles, args.output)