from __future__ import annotations

import json
import random
import wave
from pathlib import Path

import numpy as np
import torch


class AmbientNoiseBank:
    def __init__(self, data_root: str | Path, expected_channels: int = 8) -> None:
        self.data_root = Path(data_root)
        self.expected_channels = expected_channels
        self.noise_wavs = self._find_ambient_wavs()

    def _find_ambient_wavs(self) -> list[Path]:
        wavs: list[Path] = []
        for label_path in self.data_root.glob("*/label.json"):
            with label_path.open("r", encoding="utf-8") as fp:
                label = json.load(fp)
            source = label.get("drone", {}).get("sound_source")
            if source == "Ambient Noise":
                wav_path = label_path.with_name("output.wav")
                if wav_path.exists():
                    wavs.append(wav_path)
        return sorted(wavs)

    @staticmethod
    def _pcm_bytes_to_float(frame_bytes: bytes, sample_width: int) -> np.ndarray:
        if sample_width == 4:
            audio = np.frombuffer(frame_bytes, dtype="<i4").astype(np.float32)
            return audio / float(2**31)
        if sample_width == 2:
            audio = np.frombuffer(frame_bytes, dtype="<i2").astype(np.float32)
            return audio / float(2**15)
        if sample_width == 1:
            audio = np.frombuffer(frame_bytes, dtype=np.uint8).astype(np.float32)
            return (audio - 128.0) / 128.0
        raise RuntimeError(f"Unsupported sample width: {sample_width}")

    def sample_segment(self, num_frames: int, device: torch.device) -> torch.Tensor:
        if not self.noise_wavs:
            return torch.zeros((self.expected_channels, num_frames), dtype=torch.float32, device=device)

        wav_path = random.choice(self.noise_wavs)
        with wave.open(str(wav_path), "rb") as wav:
            channels = wav.getnchannels()
            n_frames = wav.getnframes()
            sample_width = wav.getsampwidth()
            if channels != self.expected_channels:
                raise RuntimeError(
                    f"Expected {self.expected_channels} channels, got {channels} in {wav_path}"
                )

            if n_frames <= num_frames:
                start = 0
            else:
                start = random.randint(0, n_frames - num_frames)
            wav.setpos(start)
            frame_bytes = wav.readframes(num_frames)

        audio = self._pcm_bytes_to_float(frame_bytes, sample_width)
        if audio.size % self.expected_channels != 0:
            audio = audio[: audio.size - (audio.size % self.expected_channels)]
        audio = audio.reshape(-1, self.expected_channels).T  # [C, T]

        if audio.shape[1] < num_frames:
            pad = np.zeros((self.expected_channels, num_frames - audio.shape[1]), dtype=np.float32)
            audio = np.concatenate([audio, pad], axis=1)
        return torch.from_numpy(audio.copy()).to(device=device, dtype=torch.float32)


class WaveformAugmenter:
    """
    Physics-aware waveform augmentation:
    - random mic dropout
    - ambient noise mixing
    - circular channel rotation on the two 4-mic rings
    """

    def __init__(
        self,
        data_root: str | Path,
        mic_dropout_prob: float = 0.25,
        mic_dropout_max: int = 1,
        ambient_noise_prob: float = 0.30,
        ambient_snr_db_min: float = 5.0,
        ambient_snr_db_max: float = 20.0,
        global_gain_prob: float = 0.30,
        global_gain_db_max: float = 6.0,
        per_channel_gain_prob: float = 0.15,
        per_channel_gain_db_max: float = 1.5,
        circular_shift_prob: float = 0.0,
    ) -> None:
        self.mic_dropout_prob = mic_dropout_prob
        self.mic_dropout_max = mic_dropout_max
        self.ambient_noise_prob = ambient_noise_prob
        self.ambient_snr_db_min = ambient_snr_db_min
        self.ambient_snr_db_max = ambient_snr_db_max
        self.global_gain_prob = global_gain_prob
        self.global_gain_db_max = global_gain_db_max
        self.per_channel_gain_prob = per_channel_gain_prob
        self.per_channel_gain_db_max = per_channel_gain_db_max
        self.circular_shift_prob = circular_shift_prob
        self.noise_bank = AmbientNoiseBank(data_root=data_root, expected_channels=8)

    @staticmethod
    def _rotate_rings(sample: torch.Tensor, steps: int) -> torch.Tensor:
        # Channel layout: lower ring [0..3], upper ring [4..7].
        out = sample.clone()
        out[0:4] = torch.roll(sample[0:4], shifts=steps, dims=0)
        out[4:8] = torch.roll(sample[4:8], shifts=steps, dims=0)
        return out

    def _mix_ambient_noise(self, signal: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        snr_db = random.uniform(self.ambient_snr_db_min, self.ambient_snr_db_max)
        signal_rms = signal.pow(2).mean().sqrt().clamp_min(1e-8)
        noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-8)
        gain = signal_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
        return signal + noise * gain

    def _apply_gain_jitter(self, signal: torch.Tensor) -> torch.Tensor:
        out = signal
        if random.random() < self.global_gain_prob:
            gain_db = random.uniform(-self.global_gain_db_max, self.global_gain_db_max)
            out = out * (10.0 ** (gain_db / 20.0))

        if random.random() < self.per_channel_gain_prob:
            gain_db = torch.empty((8, 1), device=out.device).uniform_(
                -self.per_channel_gain_db_max, self.per_channel_gain_db_max
            )
            out = out * torch.pow(10.0, gain_db / 20.0)
        return out

    def __call__(
        self,
        audio: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # audio: [B, 8, T]
        aug_audio = audio.clone()
        aug_targets = {k: v.clone() for k, v in targets.items()}
        bsz, _, num_frames = aug_audio.shape

        for idx in range(bsz):
            if random.random() < self.mic_dropout_prob:
                n_drop = random.randint(1, max(1, self.mic_dropout_max))
                drop_idx = random.sample(range(8), k=min(n_drop, 8))
                aug_audio[idx, drop_idx] = 0.0

            if random.random() < self.ambient_noise_prob:
                noise = self.noise_bank.sample_segment(num_frames=num_frames, device=aug_audio.device)
                aug_audio[idx] = self._mix_ambient_noise(aug_audio[idx], noise)

            aug_audio[idx] = self._apply_gain_jitter(aug_audio[idx])

            # Assumes each +1 ring shift corresponds to +90 deg microphone frame rotation.
            if random.random() < self.circular_shift_prob:
                steps = random.randint(1, 3)
                aug_audio[idx] = self._rotate_rings(aug_audio[idx], steps=steps)
                aug_targets["azimuth_deg"][idx] = torch.remainder(
                    aug_targets["azimuth_deg"][idx] - (steps * 90.0), 360.0
                )
                aug_targets["side_class"][idx] = torch.remainder(
                    aug_targets["side_class"][idx] - steps, 4
                )
        return aug_audio, aug_targets
