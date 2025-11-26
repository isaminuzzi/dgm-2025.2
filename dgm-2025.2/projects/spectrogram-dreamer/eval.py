import os
import shutil
import argparse
import pandas as pd
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# --- Imports ---
from infer import StyleTransferInference, log_mel_to_waveform
from src.preprocessing.generate_spectrogram import AudioFile
from frechet_audio_distance import FrechetAudioDistance
from mel_cepstral_distance import compare_audio_files
from src.utils.logger import get_logger

_logger = get_logger("eval_pipeline")

# ==========================================
# 1. VISUALIZATION HELPER
# ==========================================
def save_comparison_plot(orig_spec, gen_spec, save_path):
    """
    Saves a side-by-side comparison of Log-Mel Spectrograms.
    Args:
        orig_spec: Tensor (n_mels, time)
        gen_spec: Tensor (n_mels, time)
        save_path: Path object or string
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original
    orig_vis = orig_spec.cpu().numpy()
    ax[0].imshow(orig_vis, origin='lower', aspect='auto', cmap='magma')
    ax[0].set_title("Input (Ground Truth)")
    ax[0].axis('off')
    
    # Generated
    gen_vis = gen_spec.cpu().numpy()
    ax[1].imshow(gen_vis, origin='lower', aspect='auto', cmap='magma')
    ax[1].set_title("Output (DreamerV2 Generated)")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig) # Close to free memory

# ==========================================
# 2. METRICS
# ==========================================
class FAD_Wrapper:
    def __init__(self, gt_dir, gen_dir):
        self.gt_dir = str(gt_dir)
        self.gen_dir = str(gen_dir)
        self.frechet = FrechetAudioDistance(
            model_name="vggish", sample_rate=16000, 
            use_pca=False, use_activation=False, verbose=False
        )
    
    def evaluate(self):
        print("Calculating FAD...")
        try:
            return self.frechet.score(self.gt_dir, self.gen_dir, dtype="float32")
        except Exception as e:
            print(f"FAD Failed: {e}")
            return float('nan')

class MCD_Wrapper:
    def __init__(self, gt_dir, gen_dir):
        self.gt_dir = Path(gt_dir)
        self.gen_dir = Path(gen_dir)
        self.ext = ".wav"

    def evaluate(self):
        gt_files = sorted(list(self.gt_dir.glob(f"*{self.ext}")))
        gen_files = sorted(list(self.gen_dir.glob(f"*{self.ext}")))
        gen_map = {f.name: f for f in gen_files}
        results = []
        
        print("Calculating MCD...")
        for gt in tqdm(gt_files):
            if gt.name in gen_map:
                try:
                    mcd, penalty = compare_audio_files(str(gt), str(gen_map[gt.name]))
                    results.append({"filename": gt.name, "mcd": mcd, "penalty": penalty})
                except Exception:
                    pass 
        return pd.DataFrame(results)

# ==========================================
# 3. INFERENCE
# ==========================================
class BatchInference(StyleTransferInference):
    def generate_only(self, audio_path, styles_path):
        """
        Returns:
            wav_orig, wav_gen, spec_orig, spec_gen
        """
        target_sr = 22050
        
        # 1. Load & Resample
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            # Ensure (1, T)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
                
        except Exception as e:
            return None, None, None, None

        # Save temp for AudioFile class
        temp_path = "temp_eval.wav"
        torchaudio.save(temp_path, waveform, target_sr)
        
        try:
            # 2. Segment
            audio = AudioFile(temp_path, n_fft=512, win_length=20, hop_length=10, 
                              n_mels=80, f_min=50, f_max=7600, segment_duration=0.1, overlap=0.0)
            
            raw_segments = audio.segment_spectrogram()
            style_tensor = self.load_styles(styles_path)
            
            # 3. Check Alignment
            n_steps = min(len(raw_segments), len(style_tensor))
            if n_steps == 0:
                return None, None, None, None

            # 4. Model Forward
            obs_input = torch.stack(raw_segments[:n_steps]).to(self.device).unsqueeze(0).unsqueeze(2)
            act_input = style_tensor[:n_steps].to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(obs_input, act_input, compute_loss=False)
            
            # 5. Reconstruct
            reconstructed = outputs['reconstructed'].squeeze(0).squeeze(1)
            
            # STITCH SPECTROGRAMS FOR PLOTTING
            gen_spec = torch.cat([seg for seg in reconstructed], dim=1)
            orig_spec = torch.cat(raw_segments[:n_steps], dim=1)
            
            # 6. Vocode
            wav_gen = log_mel_to_waveform(gen_spec, audio.n_fft, audio.win_length, audio.hop_length, 
                                          audio.sample_rate, 80, audio.f_min, audio.f_max)
            
            if wav_gen.dim() == 1:
                wav_gen = wav_gen.unsqueeze(0)
            
            # Crop to match
            min_len = min(waveform.shape[1], wav_gen.shape[1])
            
            return (waveform[:, :min_len], 
                    wav_gen.cpu()[:, :min_len], 
                    orig_spec, 
                    gen_spec)

        finally:
            if os.path.exists(temp_path): os.remove(temp_path)

# ==========================================
# 4. MAIN
# ==========================================
def get_dataset(audio_dir, styles_dir, limit=100):
    audio_path = Path(audio_dir)
    styles_path = Path(styles_dir)
    pairs = []
    
    print("Scanning style folders...")
    style_folders = sorted([d for d in styles_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(style_folders)} style folders. Matching with audio...")
    
    for s_folder in style_folders:
        file_id = s_folder.name
        if (audio_path / f"{file_id}.mp3").exists():
            pairs.append((audio_path / f"{file_id}.mp3", s_folder))
        elif (audio_path / f"{file_id}.wav").exists():
            pairs.append((audio_path / f"{file_id}.wav", s_folder))
            
        if len(pairs) >= limit:
            break
            
    return pairs

def run_evaluation(model_path, audio_dir, styles_dir, output_dir, samples):
    root = Path(output_dir)
    gt_dir = root / "ground_truth"
    gen_dir = root / "generated"
    spec_dir = root / "spectrograms" # New folder for images
    
    # Cleanup & Create Dirs
    for d in [gt_dir, gen_dir, spec_dir]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    
    # 1. Get Data
    dataset = get_dataset(audio_dir, styles_dir, samples)
    print(f"Selected {len(dataset)} pairs for evaluation.")
    
    if not dataset:
        print("Error: No matching pairs found.")
        return

    # 2. Load Model
    inferencer = BatchInference(model_path, action_size=53, n_mels=80)
    
    # 3. Generate
    print("--- Running Generation ---")
    count = 0
    for audio_p, style_p in tqdm(dataset):
        # Unpack 4 return values
        orig_wav, gen_wav, orig_spec, gen_spec = inferencer.generate_only(str(audio_p), str(style_p))
        
        if orig_wav is not None and gen_wav is not None:
            # Save Audio
            torchaudio.save(str(gt_dir / f"{audio_p.stem}.wav"), orig_wav, 22050)
            torchaudio.save(str(gen_dir / f"{audio_p.stem}.wav"), gen_wav, 22050)
            
            # Save Spectrogram Plot
            plot_path = spec_dir / f"{audio_p.stem}.png"
            save_comparison_plot(orig_spec, gen_spec, plot_path)
            
            count += 1
    
    print(f"Generated {count} valid audio/spectrogram pairs.")
    if count == 0: return

    # 4. Metrics
    print("--- Calculating Metrics ---")
    fad = FAD_Wrapper(gt_dir, gen_dir).evaluate()
    df_mcd = MCD_Wrapper(gt_dir, gen_dir).evaluate()
    
    # 5. Save
    if df_mcd.empty:
        mcd_mean, mcd_std, pen_mean = 0, 0, 0
    else:
        mcd_mean = df_mcd.mcd.mean()
        mcd_std = df_mcd.mcd.std()
        pen_mean = df_mcd.penalty.mean()

    summary = pd.DataFrame({
        "metric": ["FAD", "MCD_Mean", "MCD_Std", "Penalty"],
        "value": [fad, mcd_mean, mcd_std, pen_mean]
    })
    
    print("\nRESULTS:")
    print(summary)
    
    summary.to_csv(root / "summary.csv", index=False)
    if not df_mcd.empty:
        df_mcd.to_csv(root / "mcd_detailed.csv", index=False)
    print(f"\nSaved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--styles_dir", required=True)
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    
    run_evaluation(args.model, args.audio_dir, args.styles_dir, args.output_dir, args.samples)