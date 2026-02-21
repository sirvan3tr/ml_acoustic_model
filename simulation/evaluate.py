import torch
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from acoustic_ml.data import extract_mel_spectrogram

def evaluate_flight(model_path: str, model_class, audio_path: str, trajectory: np.ndarray, trajectory_times: np.ndarray, sr: int = 96000, window_sec: float = 1.0, hop_sec: float = 0.5):
    """
    Runs a trained model on a synthetic flight audio file and plots the predicted vs true trajectories.
    Assumes standard 8-channel Mel spectrogram features.
    """
    device = torch.device('cuda' if torch.cuda.is_axis_available() else 'cpu')
    print(f"Loading model on {device}...")
    
    # Load model
    model = model_class()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint a {model_path} not found.")
        
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    # Load synthetic audio (8 channels)
    print("Loading synthetic audio...")
    audio, _ = librosa.load(audio_path, sr=sr, mono=False)
    
    if audio.ndim == 1:
        raise ValueError("Provided audio is mono, expected 8-channel.")
        
    total_samples = audio.shape[1]
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    
    num_windows = (total_samples - window_samples) // hop_samples + 1
    
    predictions = {
        'time': [],
        'distance': [],
        'height': [],
        'azimuth': []
    }
    
    true_traj = {
        'distance': [],
        'height': [],
        'azimuth': []
    }
    
    print("Running inference over windows...")
    with torch.no_grad():
        for i in tqdm(range(num_windows)):
            start_idx = i * hop_samples
            end_idx = start_idx + window_samples
            
            # The time at the center of this window
            window_center_time = (start_idx + end_idx) / 2.0 / sr
            
            # Extract features for this window
            window_audio = audio[:, start_idx:end_idx]
            features = extract_mel_spectrogram(
                torch.from_numpy(window_audio), 
                sample_rate=sr, 
                target_sr=44100, # Assuming baseline config
                n_fft=1024,
                hop_length=256,
                n_mels=256
            )
            
            # [1, 8, Mels, Time]
            batch = features.unsqueeze(0).to(device)
            
            # Inference
            output = model(batch)
            
            # Assume output structure: [distance, height, azimuth] or similar depending on model
            # For multitask_resnet: 
            # outputs["regression"] = [B, 3] -> (dist, height, azimuth_deg)
            # or it might output separate heads. We need to adapt this based on the exactly loaded model.
            
            # To handle both general cases, let's look at the output dict
            if isinstance(output, dict):
                reg = output.get("regression", None)
                if reg is not None:
                    dist_pred = reg[0, 0].item()
                    height_pred = reg[0, 1].item()
                    az_pred = reg[0, 2].item()
                else:
                    # Fallback for models outputting separate
                    dist_pred = output['distance'][0].item() if 'distance' in output else 0.0
                    height_pred = output['height'][0].item() if 'height' in output else 0.0
                    az_pred = output['azimuth'][0].item() if 'azimuth' in output else 0.0
                    
            elif isinstance(output, torch.Tensor):
                # Simple single output tensor [B, 3]
                dist_pred = output[0, 0].item()
                height_pred = output[0, 1].item()
                az_pred = output[0, 2].item()
            else:
                dist_pred = height_pred = az_pred = 0.0
                
            predictions['time'].append(window_center_time)
            predictions['distance'].append(dist_pred)
            predictions['height'].append(height_pred)
            predictions['azimuth'].append(az_pred)
            
            # Get closest ground truth
            idx = np.argmin(np.abs(trajectory_times - window_center_time))
            pos = trajectory[idx] # [x, y, z]
            
            true_dist = np.linalg.norm(pos)
            true_height = pos[2]
            true_az = np.rad2deg(np.arctan2(pos[1], pos[0]))
            if true_az < 0:
                true_az += 360
                
            true_traj['distance'].append(true_dist)
            true_traj['height'].append(true_height)
            true_traj['azimuth'].append(true_az)
            
    # Plot results
    plt.figure(figsize=(15, 10))
    
    times = predictions['time']
    
    plt.subplot(3, 1, 1)
    plt.plot(times, true_traj['distance'], 'k--', label='True Distance')
    plt.plot(times, predictions['distance'], 'b-', label='Predicted Distance')
    plt.ylabel('Distance (m)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(times, true_traj['height'], 'k--', label='True Height')
    plt.plot(times, predictions['height'], 'g-', label='Predicted Height')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(times, true_traj['azimuth'], 'k--', label='True Azimuth')
    plt.plot(times, predictions['azimuth'], 'r-', label='Predicted Azimuth')
    plt.ylabel('Azimuth (deg)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(audio_path.replace('.wav', '_eval.png'))
    print(f"Plot saved to {audio_path.replace('.wav', '_eval.png')}")

if __name__ == "__main__":
    import sys
    # Example usage: python evaluate.py path/to/model.pt path/to/audio.wav
    pass # Will build robust runner 
