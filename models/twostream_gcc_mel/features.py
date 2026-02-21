from __future__ import annotations

from itertools import combinations

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
    return torch.maximum(torch.zeros(1, dtype=dtype), torch.minimum(lower, upper))


def all_mic_pairs(num_channels: int) -> list[tuple[int, int]]:
    return list(combinations(range(num_channels), 2))


class TwoStreamFeatureBuilder:
    """
    Builds both streams from waveform [B, C, T]:
    - mel: [B, C, H, W]
    - gcc: [B, num_pairs, 2*max_lag+1]
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 256,
        mel_size: tuple[int, int] = (256, 256),
        max_lag: int = 64,
        num_channels: int = 8,
        include_gcc_std: bool = True,
        mic_pairs: list[tuple[int, int]] | None = None,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_size = mel_size
        self.max_lag = max_lag
        self.num_channels = num_channels
        self.include_gcc_std = include_gcc_std
        self.mic_pairs = mic_pairs if mic_pairs is not None else all_mic_pairs(num_channels)

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
                sample_rate=sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, dtype=dtype
            ).to(device)
        return self._mel_cache[key]

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        bsz, channels, time = audio.shape
        x = audio.reshape(bsz * channels, time)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self._window(audio.device, audio.dtype),
            center=True,
            return_complex=True,
        )
        return stft.reshape(bsz, channels, stft.shape[1], stft.shape[2])

    def build(self, audio: torch.Tensor, sample_rate: int) -> dict[str, torch.Tensor]:
        x = audio.float()
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, T], got shape={tuple(x.shape)}")
        if x.shape[1] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {x.shape[1]}")

        stft = self._stft(x)  # [B, C, F, Frames]
        mel = self._build_mel(stft, sample_rate)
        gcc = self._build_gcc_phat(stft)
        channel_energy = self._build_channel_energy(x)
        return {"mel": mel, "gcc": gcc, "channel_energy": channel_energy}

    def _build_mel(self, stft: torch.Tensor, sample_rate: int) -> torch.Tensor:
        power = stft.abs().pow(2.0)
        mel_fb = self._mel_fb(sample_rate, stft.device, stft.real.dtype)
        mel = torch.einsum("mf,bcft->bcmt", mel_fb, power).clamp_min(1e-8).log()
        mel = F.interpolate(mel, size=self.mel_size, mode="bilinear", align_corners=False)

        mean = mel.mean(dim=(1, 2, 3), keepdim=True)
        std = mel.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        return (mel - mean) / std

    def _build_gcc_phat(self, stft: torch.Tensor) -> torch.Tensor:
        center = self.n_fft // 2
        start = max(0, center - self.max_lag)
        end = min(self.n_fft, center + self.max_lag + 1)

        gcc_mean_list: list[torch.Tensor] = []
        gcc_std_list: list[torch.Tensor] = []
        for mic_a, mic_b in self.mic_pairs:
            cross = stft[:, mic_a] * torch.conj(stft[:, mic_b])  # [B, F, Frames]
            phat = cross / cross.abs().clamp_min(1e-8)
            gcc_full = torch.fft.irfft(phat, n=self.n_fft, dim=1)
            gcc_full = torch.roll(gcc_full, shifts=center, dims=1)  # zero-lag centered
            gcc_window = gcc_full[:, start:end, :]  # [B, L, Frames]
            gcc_mean = gcc_window.mean(dim=2)  # [B, L]
            gcc_mean_list.append(gcc_mean)
            if self.include_gcc_std:
                gcc_std = gcc_window.std(dim=2).clamp_min(1e-6)
                gcc_std_list.append(gcc_std)

        gcc_features = torch.stack(gcc_mean_list, dim=1)  # [B, Pairs, L]
        if self.include_gcc_std:
            gcc_std_features = torch.stack(gcc_std_list, dim=1)
            gcc_features = torch.cat([gcc_features, gcc_std_features], dim=1)
        mean = gcc_features.mean(dim=2, keepdim=True)
        std = gcc_features.std(dim=2, keepdim=True).clamp_min(1e-5)
        return (gcc_features - mean) / std

    def _build_channel_energy(self, audio: torch.Tensor) -> torch.Tensor:
        # [B, C]
        rms = audio.pow(2).mean(dim=2).sqrt().clamp_min(1e-8).log()
        rms = rms - rms.mean(dim=1, keepdim=True)
        rms = rms / rms.std(dim=1, keepdim=True).clamp_min(1e-5)
        return rms
