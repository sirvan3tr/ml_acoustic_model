"""
simulate_v4.py

End-to-end drone flight simulation + v4 model evaluation.

Usage:
    PYTHONPATH=. uv run simulation/simulate_v4.py \
        --audio mic_array_data/20241115_093128/output.wav \
        --model models/multitask_resnet_kendall_v4/runs/default/best.pt \
        --trajectory orbit \
        --duration 30
"""

import argparse
import math
import sys
import os
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa

# ─────────────────────────────────────────────────
# 1. Virtual Mic Array Environment
# ─────────────────────────────────────────────────

SPEED_OF_SOUND = 343.0

class VirtualMicArray:
    """Replicates the 8-mic array geometry from the dataset label.json files."""
    def __init__(self):
        self.radius = 1.72
        lower_h, upper_h = 1.49, 2.38
        lower_az = [0.0, 90.0, 180.0, 270.0]
        upper_az = [45.0, 135.0, 225.0, 315.0]

        self.positions = []   # (3,) per mic
        self.directions = []  # (3,) unit vector the mic points in
        for az in lower_az:
            self.positions.append(self._az_to_xy(az, lower_h))
            self.directions.append(self._dir_vector(az, el=10.0))
        for az in upper_az:
            self.positions.append(self._az_to_xy(az, upper_h))
            self.directions.append(self._dir_vector(az, el=20.0))

        self.positions = np.array(self.positions)   # (8,3)
        self.directions = np.array(self.directions) # (8,3)

    def _az_to_xy(self, az_deg, z):
        r = np.deg2rad(az_deg)
        return np.array([self.radius * np.cos(r), self.radius * np.sin(r), z])

    def _dir_vector(self, az_deg, el=10.0):
        r_az = np.deg2rad(az_deg)
        r_el = np.deg2rad(el)
        return np.array([
            np.cos(r_el) * np.cos(r_az),
            np.cos(r_el) * np.sin(r_az),
            np.sin(r_el),
        ])


def compute_propagation(source_pos, array: VirtualMicArray):
    """Returns delays (s) and amplitude attenuations per mic."""
    vecs = source_pos - array.positions        # (8,3)
    dists = np.linalg.norm(vecs, axis=1)      # (8,)
    safe_d = np.clip(dists, 0.1, None)

    delays = safe_d / SPEED_OF_SOUND

    # Supercardioid polar response: 0.37 + 0.63*cos(θ)
    unit_vecs = vecs / safe_d[:, np.newaxis]
    cos_theta = np.sum(array.directions * unit_vecs, axis=1)
    polar = np.abs(0.37 + 0.63 * cos_theta)
    attens = (1.0 / safe_d) * polar

    return delays, attens


# ─────────────────────────────────────────────────
# 2. Trajectory Generation
# ─────────────────────────────────────────────────

def orbit(radius, height, speed, duration, ctrl_rate=100):
    t = np.arange(0, duration, 1.0 / ctrl_rate)
    omega = speed / radius
    x = radius * np.cos(omega * t)
    y = radius * np.sin(omega * t)
    z = np.full_like(x, height)
    return t, np.stack([x, y, z], axis=1)


def spiral(r_min=5.0, r_max=50.0, height=10.0, n_revolutions=2.0, duration=60.0, ctrl_rate=100):
    """
    Drone spirals outward from r_min to r_max at constant height.
    The radius increases linearly with time.
    """
    t = np.arange(0, duration, 1.0 / ctrl_rate)
    frac = t / duration
    radius_t = r_min + (r_max - r_min) * frac
    # n_revolutions full circles over the duration
    angle_t = frac * n_revolutions * 2.0 * np.pi
    x = radius_t * np.cos(angle_t)
    y = radius_t * np.sin(angle_t)
    z = np.full_like(x, height)
    return t, np.stack([x, y, z], axis=1)


def flyby(x_start=-50.0, x_end=50.0, y_offset=5.0, height=15.0, duration=30.0, ctrl_rate=100):
    t = np.arange(0, duration, 1.0 / ctrl_rate)
    frac = t / duration
    x = x_start + (x_end - x_start) * frac
    y = np.full_like(x, y_offset)
    z = np.full_like(x, height)
    return t, np.stack([x, y, z], axis=1)


# ─────────────────────────────────────────────────
# 3. Audio Synthesis
# ─────────────────────────────────────────────────

def fractional_delay(chunk, delay_samples):
    """Linear-interpolation fractional delay."""
    d_int = int(np.floor(delay_samples))
    d_frac = delay_samples - d_int
    padded = np.pad(chunk, (0, d_int + 2))
    out = padded[d_int: d_int + len(chunk)] * (1.0 - d_frac) + \
          padded[d_int + 1: d_int + 1 + len(chunk)] * d_frac
    return out


def synthesize(base_mono, sr, array, traj_pos, traj_t, chunk_dur=0.05):
    """Produce (8, T) 8-channel audio from mono source + trajectory."""
    n_mics = len(array.positions)
    T = len(base_mono)
    out = np.zeros((n_mics, T), dtype=np.float32)
    chunk_n = int(chunk_dur * sr)

    for i in tqdm(range(0, T, chunk_n), desc="Synthesizing"):
        chunk = base_mono[i: i + chunk_n]
        if len(chunk) == 0:
            break
        center_t = (i + min(i + chunk_n, T)) / 2.0 / sr
        idx = np.argmin(np.abs(traj_t - center_t))
        pos = traj_pos[idx]
        delays_s, attens = compute_propagation(pos, array)
        for m in range(n_mics):
            d_samp = delays_s[m] * sr
            delayed = fractional_delay(chunk, d_samp)
            end = min(i + len(chunk), T)
            out[m, i:end] += delayed[: end - i] * attens[m]

    return out


# ─────────────────────────────────────────────────
# 4. v4 Feature Extraction (inline, matching features.py exactly)
# ─────────────────────────────────────────────────

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def make_mel_fb_torch(sample_rate, n_fft, n_mels, device):
    f_max = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, f_max, n_freqs, device=device)
    mel_min = _hz_to_mel(0.0)
    mel_max = _hz_to_mel(f_max)
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    ramps = hz_pts.unsqueeze(1) - fft_freqs.unsqueeze(0)
    denom = (hz_pts[1:] - hz_pts[:-1]).clamp_min(1e-6)
    lower = -ramps[:-2] / denom[:-1].unsqueeze(1)
    upper =  ramps[2:]  / denom[1:].unsqueeze(1)
    return torch.maximum(torch.zeros(1, device=device), torch.minimum(lower, upper))


def extract_features_v4(audio_8ch, sr, n_fft=1024, hop_length=512, n_mels=256, out_size=(256, 256)):
    """
    audio_8ch: torch.Tensor [8, T]
    Returns: [24, 256, 256] (8 mel + 8 rel + 8 ipd)
    """
    device = audio_8ch.device
    x = audio_8ch.float()
    window = torch.hann_window(n_fft, periodic=True, device=device)

    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, center=True, return_complex=True,
    )  # [8, F, Frames]

    mel_fb = make_mel_fb_torch(sr, n_fft, n_mels, device)  # [n_mels, F]
    power  = spec.abs().pow(2.0)
    mel    = torch.einsum("mf,cft->cmt", mel_fb, power)
    mel    = torch.log(mel.clamp_min(1e-8))
    mel    = F.interpolate(mel.unsqueeze(0), size=out_size, mode="bilinear", align_corners=False).squeeze(0)

    mean, std = mel.mean(), mel.std().clamp_min(1e-5)
    mel = (mel - mean) / std

    mel_rel = mel - mel.mean(dim=0, keepdim=True)

    phase     = spec.angle()
    sin_p     = torch.sin(phase)
    cos_p     = torch.cos(phase)
    mel_sin   = torch.einsum("mf,cft->cmt", mel_fb, sin_p)
    mel_cos   = torch.einsum("mf,cft->cmt", mel_fb, cos_p)
    mel_phase = torch.atan2(mel_sin, mel_cos)
    mel_phase = F.interpolate(mel_phase.unsqueeze(0), size=out_size, mode="bilinear", align_corners=False).squeeze(0)
    mean_phase = torch.atan2(
        torch.sin(mel_phase).mean(dim=0, keepdim=True),
        torch.cos(mel_phase).mean(dim=0, keepdim=True),
    )
    ipd      = torch.remainder(mel_phase - mean_phase + math.pi, 2 * math.pi) - math.pi
    ipd_norm = ipd / math.pi

    return torch.cat([mel, mel_rel, ipd_norm], dim=0)  # [24, 256, 256]


# ─────────────────────────────────────────────────
# 5. v4 Model (identical to model.py)
# ─────────────────────────────────────────────────

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multitask_resnet_kendall_v4.model import MultiTaskResNet


# ─────────────────────────────────────────────────
# 6. Ground-truth helpers
# ─────────────────────────────────────────────────

def pos_to_gt(pos):
    """Convert [x,y,z] position to (distance_m, height_m, azimuth_deg)."""
    dist   = float(np.linalg.norm(pos[:2]))   # planar distance
    height = float(pos[2])
    az     = float(np.degrees(np.arctan2(pos[1], pos[0])))
    if az < 0:
        az += 360.0
    return dist, height, az


# ─────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",        required=True,
                        help="Path to an 8-ch wav — use a real Drone recording from the dataset")
    parser.add_argument("--model",        required=True, help="Path to best.pt for v4 model")
    parser.add_argument("--trajectory",   choices=["orbit", "spiral", "flyby"], default="spiral")
    parser.add_argument("--duration",     type=float, default=60.0)
    parser.add_argument("--r_min",        type=float, default=5.0,  help="Min radius (m)")
    parser.add_argument("--r_max",        type=float, default=50.0, help="Max radius (m)")
    parser.add_argument("--height",       type=float, default=10.0, help="Flight height (m)")
    parser.add_argument("--n_revolutions",type=float, default=2.0,  help="Revolutions for spiral")
    parser.add_argument("--out_dir",      default="simulation/outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sr = 96000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load base audio ──
    # We take one channel from a real *Drone* recording as the mono sound source.
    # This gives the synthesized audio realistic drone acoustics.
    print(f"Loading base audio (drone recording): {args.audio}")
    raw, file_sr = librosa.load(args.audio, sr=sr, mono=False)
    if raw.ndim > 1:
        # Use a middle section — skip first 5s (engine spool-up) and last 5s
        skip = int(5.0 * sr)
        raw = raw[:, skip:-skip] if raw.shape[1] > 2 * skip else raw
        mono = raw[0]   # take channel 1
    else:
        mono = raw
    n_target = int(args.duration * sr)
    if len(mono) < n_target:
        mono = np.tile(mono, math.ceil(n_target / len(mono)))
    mono = (mono[:n_target] / (np.abs(mono[:n_target]).max() + 1e-8) * 0.5).astype(np.float32)

    # ── Generate trajectory ──
    print(f"Generating trajectory: {args.trajectory}  "
          f"(r={args.r_min}–{args.r_max} m, h={args.height} m)")
    if args.trajectory == "orbit":
        traj_t, traj_pos = orbit(
            radius=args.r_min, height=args.height, speed=2.0, duration=args.duration
        )
    elif args.trajectory == "spiral":
        traj_t, traj_pos = spiral(
            r_min=args.r_min, r_max=args.r_max, height=args.height,
            n_revolutions=args.n_revolutions, duration=args.duration
        )
    else:
        traj_t, traj_pos = flyby(
            x_start=-args.r_max, x_end=args.r_max,
            height=args.height, duration=args.duration
        )

    # ── Synthesize ──
    array = VirtualMicArray()
    out_8ch = synthesize(mono, sr, array, traj_pos, traj_t)

    out_wav = os.path.join(args.out_dir, f"sim_{args.trajectory}.wav")
    sf.write(out_wav, out_8ch.T, sr)
    print(f"Saved 8-ch wav: {out_wav}")

    # ── Load model ──
    print("Loading v4 model …")
    model = MultiTaskResNet(in_channels=24).to(device)
    ckpt  = torch.load(args.model, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # ── Inference (sliding 1.5-s windows, 0.75-s hop) ──
    window_s = 1.5
    hop_s    = 0.75
    win_n    = int(window_s * sr)
    hop_n    = int(hop_s    * sr)
    T        = out_8ch.shape[1]
    n_wins   = max(1, (T - win_n) // hop_n + 1)

    pred_dist, pred_h, pred_az = [], [], []
    gt_dist,   gt_h,   gt_az  = [], [], []
    times = []

    print("Running inference …")
    audio_tensor = torch.from_numpy(out_8ch).to(device)  # [8, T]

    with torch.no_grad():
        for i in tqdm(range(n_wins)):
            s = i * hop_n
            e = s + win_n
            if e > T:
                break
            chunk = audio_tensor[:, s:e]  # [8, win_n]

            feat = extract_features_v4(chunk, sr)   # [24, 256, 256]
            batch = feat.unsqueeze(0)               # [1, 24, 256, 256]

            out   = model(batch)
            d_hat = out["distance"][0].item()
            h_hat = out["height"][0].item()
            # azimuth head outputs [sin, cos] — convert to degrees
            az_xy = out["azimuth_xy"][0]   # [2]
            az_hat = math.degrees(math.atan2(az_xy[0].item(), az_xy[1].item()))
            if az_hat < 0:
                az_hat += 360.0

            center_t = (s + e) / 2.0 / sr
            idx = int(np.argmin(np.abs(traj_t - center_t)))
            gd, gh, ga = pos_to_gt(traj_pos[idx])

            pred_dist.append(d_hat);  gt_dist.append(gd)
            pred_h.append(h_hat);     gt_h.append(gh)
            pred_az.append(az_hat);   gt_az.append(ga)
            times.append(center_t)

    times      = np.array(times)
    pred_dist  = np.array(pred_dist)
    pred_h     = np.array(pred_h)
    pred_az    = np.array(pred_az)
    gt_dist    = np.array(gt_dist)
    gt_h       = np.array(gt_h)
    gt_az      = np.array(gt_az)

    # ── Metrics ──
    mae_d  = float(np.mean(np.abs(pred_dist - gt_dist)))
    mae_h  = float(np.mean(np.abs(pred_h - gt_h)))
    az_err = np.abs(pred_az - gt_az)
    az_err = np.minimum(az_err, 360.0 - az_err)   # circular wrap
    mae_az = float(np.mean(az_err))

    print("\n" + "="*50)
    print("  SIMULATION RESULTS (v4 model)")
    print("="*50)
    print(f"  Distance MAE : {mae_d:.3f} m   (GT constant ≈ {gt_dist.mean():.1f} m)")
    print(f"  Height MAE   : {mae_h:.3f} m   (GT constant ≈ {gt_h.mean():.1f} m)")
    print(f"  Azimuth MAE  : {mae_az:.2f}°  (full 0-360 range)")
    print("="*50)

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    title = (f"v4 Model — {args.trajectory}  "
             f"r={args.r_min}–{args.r_max} m  h={args.height} m  ({args.duration}s)")
    fig.suptitle(title, fontsize=13)

    axes[0].plot(times, gt_dist,   "k--", linewidth=1.5, label="GT Distance")
    axes[0].plot(times, pred_dist, "b-",  linewidth=1.2, label=f"Predicted  (MAE={mae_d:.2f} m)")
    axes[0].set_ylabel("Distance (m)")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(times, gt_h,   "k--", linewidth=1.5, label=f"GT Height ({args.height} m)")
    axes[1].plot(times, pred_h, "g-",  linewidth=1.2, label=f"Predicted (MAE={mae_h:.2f} m)")
    axes[1].set_ylabel("Height (m)")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(times, gt_az,   "k--", linewidth=1.5, label="GT Azimuth")
    axes[2].plot(times, pred_az, "r-",  linewidth=1.2, label=f"Predicted (MAE={mae_az:.1f}°)")
    axes[2].set_ylabel("Azimuth (°)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(); axes[2].grid(True)

    out_png = os.path.join(args.out_dir, f"sim_{args.trajectory}_v4_eval.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\nPlot saved → {out_png}")


if __name__ == "__main__":
    main()
