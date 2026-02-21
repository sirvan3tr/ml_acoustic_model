import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm

from simulation.env import VirtualMicArray, compute_propagation

def load_base_audio(audio_path: str, target_sr: int = 96000) -> np.ndarray:
    """Loads a single-channel audio source"""
    # Load first channel if multi-channel
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=False)
    if audio.ndim > 1:
        audio = audio[0] # Take first channel
    # Normalize to avoid clipping later just in case
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.5
    return audio

def apply_fractional_delay(audio_chunk: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    Applies a fractional sample delay to an audio chunk using linear interpolation.
    For more accuracy, sincere interpolation or frequency domain shift could be used,
    but linear is usually sufficient and fast for high sample rates like 96kHz.
    """
    int_delay = int(np.floor(delay_samples))
    frac_delay = delay_samples - int_delay
    
    # Pad input
    padded = np.pad(audio_chunk, (0, int_delay + 2), mode='constant')
    
    # Linear interpolation
    shifted = padded[int_delay : int_delay + len(audio_chunk)] * (1 - frac_delay) + \
              padded[int_delay + 1 : int_delay + 1 + len(audio_chunk)] * frac_delay
              
    return shifted

def synthesize_flight(base_audio: np.ndarray, 
                      sr: int, 
                      array: VirtualMicArray, 
                      trajectory: np.ndarray, 
                      trajectory_times: np.ndarray) -> np.ndarray:
    """
    Synthesizes an 8-channel audio file from a mono source moving along a trajectory.
    
    Args:
        base_audio: Mono audio signal
        sr: Sample rate
        array: VirtualMicArray
        trajectory: (N, 3) positions
        trajectory_times: (N,) timestamps
        
    Returns:
        (8, T) synthetic multi-channel audio
    """
    num_mics = len(array.mics)
    total_samples = len(base_audio)
    out_audio = np.zeros((num_mics, total_samples))
    
    # Process in chunks (e.g., 0.1s chunks)
    chunk_duration = 0.05 # seconds
    chunk_samples = int(chunk_duration * sr)
    num_chunks = int(np.ceil(total_samples / chunk_samples))
    
    # Precompute sample indices
    time_points = np.arange(total_samples) / sr
    
    # Interpolate trajectory to audio sample times (for smoother processing, though we chunk)
    # We will just evaluate trajectory at the center of each chunk for simplicity
    
    for i in tqdm(range(num_chunks), desc="Synthesizing"):
        start_idx = i * chunk_samples
        end_idx = min(start_idx + chunk_samples, total_samples)
        current_chunk = base_audio[start_idx:end_idx]
        
        # Center time of this chunk
        chunk_center_time = (start_idx + end_idx) / 2.0 / sr
        
        # Find closest trajectory point
        # A more rigorous approach interpolates the exact position at chunk_center_time
        time_diffs = np.abs(trajectory_times - chunk_center_time)
        traj_idx = np.argmin(time_diffs)
        current_pos = trajectory[traj_idx]
        
        # Compute propagation (delays in seconds, linear attenuations)
        delays_sec, attens = compute_propagation(current_pos, array)
        
        # Convert delay to samples
        delays_samples = delays_sec * sr
        
        # We need to apply delay relative to the minimal delay to avoid 
        # pushing audio off the front of the chunk. 
        # Alternatively, we just shift the whole signal.
        # Let's shift relative to 0 so absolute time of flight is preserved.
        
        for m in range(num_mics):
            # Apply delay and attenuation
            delayed = apply_fractional_delay(current_chunk, delays_samples[m])
            out_audio[m, start_idx:end_idx] += delayed * attens[m]
            
    return out_audio

if __name__ == "__main__":
    from simulation.trajectory import generate_orbit
    print("Testing synthesis on fake noise...")
    sr = 96000
    duration = 5.0
    
    # Generate 5 sec of white noise
    noise = np.random.randn(int(sr * duration)).astype(np.float32)
    
    array = VirtualMicArray()
    t, pos = generate_orbit(radius=15.0, height=10.0, speed=10.0, duration=duration, sample_rate=100)
    
    out = synthesize_flight(noise, sr, array, pos, t)
    
    os.makedirs("simulation/outputs", exist_ok=True)
    sf.write("simulation/outputs/test_orbit.wav", out.T, sr)
    print("Saved test synthesis to simulation/outputs/test_orbit.wav")
