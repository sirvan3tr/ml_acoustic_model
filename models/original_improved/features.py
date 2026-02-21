from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(freq_mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (freq_mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if f_max is None:
        f_max = float(sample_rate) / 2.0

    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, f_max, n_freqs, dtype=dtype)

    mel_min = _hz_to_mel(torch.tensor(f_min, dtype=dtype))
    mel_max = _hz_to_mel(torch.tensor(f_max, dtype=dtype))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, dtype=dtype)
    hz_points = _mel_to_hz(mel_points)

    ramps = hz_points.unsqueeze(1) - fft_freqs.unsqueeze(0)
    denom = (hz_points[1:] - hz_points[:-1]).clamp_min(1e-6)
    lower = -ramps[:-2] / denom[:-1].unsqueeze(1)
    upper = ramps[2:] / denom[1:].unsqueeze(1)

    fb = torch.maximum(torch.zeros(1, dtype=dtype), torch.minimum(lower, upper))
    return fb


class OriginalImprovedMelTransform:
    """
    Original paper parameters: 44.1 kHz, 1024 n_fft, 256 hop_length, 256 mels.
    Improved by extracting and explicitly appending Inter-channel Phase Differences (IPD).
    Yields 16 channels instead of 8: [8 Mel, 8 IPD].
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 256,
        out_size: tuple[int, int] = (256, 256),
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.out_size = out_size

        self._window_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._mel_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def _window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if key not in self._window_cache:
            self._window_cache[key] = torch.hann_window(
                self.n_fft, periodic=True, device=device, dtype=dtype
            )
        return self._window_cache[key]

    def _mel_fb(self, sample_rate: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (sample_rate, device, dtype)
        if key not in self._mel_cache:
            self._mel_cache[key] = create_mel_filterbank(
                sample_rate=sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                dtype=dtype,
            ).to(device)
        return self._mel_cache[key]

    def __call__(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # audio: [C, T]
        x = audio.float()
        window = self._window(x.device, x.dtype)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        
        mel_fb = self._mel_fb(sample_rate, x.device, x.dtype)

        # 1. Take magnitude (Standard Mel Spec)
        power = spec.abs().pow(2.0)
        mel = torch.einsum("mf,cft->cmt", mel_fb, power)
        mel = torch.log(mel.clamp_min(1e-8))
        mel = F.interpolate(
            mel.unsqueeze(0), size=self.out_size, mode="bilinear", align_corners=False
        ).squeeze(0)

        mean = mel.mean()
        std = mel.std().clamp_min(1e-5)
        mel = (mel - mean) / std

        # 2. Extract Phase (IPD)
        phase = spec.angle()
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)
        
        mel_sin = torch.einsum("mf,cft->cmt", mel_fb, sin_phase)
        mel_cos = torch.einsum("mf,cft->cmt", mel_fb, cos_phase)
        mel_phase = torch.atan2(mel_sin, mel_cos)
        
        mel_phase = F.interpolate(
            mel_phase.unsqueeze(0), size=self.out_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        
        mean_phase = torch.atan2(
            torch.sin(mel_phase).mean(dim=0, keepdim=True),
            torch.cos(mel_phase).mean(dim=0, keepdim=True)
        )
        
        ipd = torch.remainder(mel_phase - mean_phase + math.pi, 2 * math.pi) - math.pi
        ipd_norm = ipd / math.pi

        # Stack into 16 channels: [8 Base, 8 IPD]
        return torch.cat([mel, ipd_norm], dim=0)
