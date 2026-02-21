from __future__ import annotations

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


class MelSpectrogramTransform:
    """
    Converts waveform [channels, time] into normalized log-mel [channels, H, W].
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 512,
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
        power = spec.abs().pow(2.0)  # [C, F, Frames]

        mel_fb = self._mel_fb(sample_rate, x.device, x.dtype)
        mel = torch.einsum("mf,cft->cmt", mel_fb, power)
        mel = torch.log(mel.clamp_min(1e-8))

        mel = F.interpolate(
            mel.unsqueeze(0), size=self.out_size, mode="bilinear", align_corners=False
        ).squeeze(0)

        # Global normalization preserves inter-channel level relationships used for direction cues.
        mean = mel.mean()
        std = mel.std().clamp_min(1e-5)
        mel = (mel - mean) / std
        return mel
