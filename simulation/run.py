import argparse
import numpy as np
import os
import torch
import importlib.util

from simulation.env import VirtualMicArray
from simulation.trajectory import generate_orbit, generate_flyby
from simulation.synthesize import load_base_audio, synthesize_flight
from simulation.evaluate import evaluate_flight
import soundfile as sf

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description="Simulate acoustic array and evaluate models.")
    parser.add_argument("--audio", type=str, required=True, help="Base mono/raw audio file to simulate moving source.")
    parser.add_argument("--model", type=str, required=True, help="Path to best model .pt file")
    parser.add_argument("--model_class_file", type=str, required=True, help="Path to python file defining the model class")
    parser.add_argument("--model_class_name", type=str, required=True, help="Name of the model class to instantiate")
    parser.add_argument("--trajectory", type=str, choices=["orbit", "flyby"], default="orbit")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration of simulation in seconds")
    parser.add_argument("--out_dir", type=str, default="simulation/outputs")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"1. Loading base audio: {args.audio}")
    sr = 96000
    base_audio = load_base_audio(args.audio, sr)
    
    # Loop/trim base audio to requested duration
    target_samples = int(args.duration * sr)
    if len(base_audio) < target_samples:
        repeats = int(np.ceil(target_samples / len(base_audio)))
        base_audio = np.tile(base_audio, repeats)[:target_samples]
    else:
        base_audio = base_audio[:target_samples]
        
    print(f"2. Generating trajectory: {args.trajectory}")
    if args.trajectory == "orbit":
        t, pos = generate_orbit(radius=15.0, height=10.0, speed=2.0, duration=args.duration, sample_rate=100)
    elif args.trajectory == "flyby":
        t, pos = generate_flyby((-20.0, 5.0, 15.0), (20.0, -5.0, 15.0), duration=args.duration, sample_rate=100)
        
    print("3. Synthesizing flight...")
    array = VirtualMicArray()
    out_audio = synthesize_flight(base_audio, sr, array, pos, t)
    
    out_wav_path = os.path.join(args.out_dir, f"simulated_{args.trajectory}.wav")
    sf.write(out_wav_path, out_audio.T, sr)
    print(f"Saved simulated 8-channel audio to {out_wav_path}")
    
    print(f"4. Loading Model Class: {args.model_class_name} from {args.model_class_file}")
    model_module = load_module_from_path("custom_model", args.model_class_file)
    ModelClass = getattr(model_module, args.model_class_name)
    
    print("5. Evaluating...")
    evaluate_flight(args.model, ModelClass, out_wav_path, pos, t, sr=sr)
    
if __name__ == "__main__":
    main()
