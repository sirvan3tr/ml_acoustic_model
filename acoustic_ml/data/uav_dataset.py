from __future__ import annotations

import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

Split = Literal["train", "test", "all"]


@dataclass(frozen=True)
class RecordingMeta:
    recording_id: str
    wav_path: Path
    label_path: Path
    source: str
    distance_m: float | None
    height_m: float | None
    azimuth_deg: float | None
    rotation: str | None
    sample_rate: int
    channels: int
    num_frames: int


@dataclass(frozen=True)
class WindowMeta:
    recording: RecordingMeta
    start_frame: int
    num_frames: int


class AcousticUAVDataset(Dataset[dict[str, object]]):
    """Reusable dataset for multi-channel UAV acoustic recordings."""

    def __init__(
        self,
        root_dir: str | Path = "mic_array_data",
        split: Split = "train",
        sources: Iterable[str] | None = ("Drone",),
        split_ratio: float = 0.75,
        window_seconds: float | None = 1.0,
        hop_seconds: float | None = None,
        transform: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        pad_short_windows: bool = True,
    ) -> None:
        if split not in {"train", "test", "all"}:
            raise ValueError(f"Unsupported split: {split}")
        if not 0.0 < split_ratio < 1.0:
            raise ValueError("split_ratio must be in (0, 1).")
        if window_seconds is not None and window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        if hop_seconds is not None and hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive.")

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        self.split = split
        self.split_ratio = split_ratio
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.transform = transform
        self.pad_short_windows = pad_short_windows
        self.source_filter = set(sources) if sources is not None else None

        self.recordings = self._index_recordings()
        if not self.recordings:
            raise RuntimeError(
                "No recordings found. Check root_dir and optional source filters."
            )

        self.distance_values = self._sorted_unique("distance_m")
        self.height_values = self._sorted_unique("height_m")
        self.azimuth_values = self._sorted_unique("azimuth_deg")
        self.rotation_values = self._rotation_order()

        self.distance_to_index = {value: idx for idx, value in enumerate(self.distance_values)}
        self.height_to_index = {value: idx for idx, value in enumerate(self.height_values)}
        self.azimuth_to_index = {value: idx for idx, value in enumerate(self.azimuth_values)}
        self.rotation_to_index = {value: idx for idx, value in enumerate(self.rotation_values)}

        self.windows = self._build_windows()
        if not self.windows:
            raise RuntimeError(
                "No sample windows were generated. Adjust split/window arguments."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        window = self.windows[index]
        recording = window.recording

        waveform = self._read_audio_window(
            wav_path=recording.wav_path,
            start_frame=window.start_frame,
            num_frames=window.num_frames,
            expected_channels=recording.channels,
            pad_short=self.pad_short_windows,
        )
        audio = torch.from_numpy(waveform.T.copy())  # [channels, time]
        if self.transform is not None:
            try:
                audio = self.transform(audio, recording.sample_rate)
            except TypeError:
                audio = self.transform(audio)

        distance = recording.distance_m if recording.distance_m is not None else math.nan
        height = recording.height_m if recording.height_m is not None else math.nan
        azimuth = recording.azimuth_deg if recording.azimuth_deg is not None else math.nan

        if math.isnan(azimuth):
            azimuth_sin = math.nan
            azimuth_cos = math.nan
        else:
            radians = math.radians(azimuth)
            azimuth_sin = math.sin(radians)
            azimuth_cos = math.cos(radians)

        targets = {
            "regression": torch.tensor([distance, height, azimuth], dtype=torch.float32),
            "azimuth_sin_cos": torch.tensor([azimuth_sin, azimuth_cos], dtype=torch.float32),
            "distance_class": torch.tensor(
                self._index_or_default(self.distance_to_index, recording.distance_m),
                dtype=torch.long,
            ),
            "height_class": torch.tensor(
                self._index_or_default(self.height_to_index, recording.height_m),
                dtype=torch.long,
            ),
            "azimuth_class": torch.tensor(
                self._index_or_default(self.azimuth_to_index, recording.azimuth_deg),
                dtype=torch.long,
            ),
            "rotation_class": torch.tensor(
                self._index_or_default(self.rotation_to_index, recording.rotation),
                dtype=torch.long,
            ),
        }

        metadata = {
            "recording_id": recording.recording_id,
            "source": recording.source,
            "sample_rate": recording.sample_rate,
            "start_frame": window.start_frame,
            "num_frames": window.num_frames,
            "wav_path": str(recording.wav_path),
        }
        return {"audio": audio, "targets": targets, "metadata": metadata}

    def _index_recordings(self) -> list[RecordingMeta]:
        recordings: list[RecordingMeta] = []
        for label_path in sorted(self.root_dir.glob("*/label.json")):
            wav_path = label_path.with_name("output.wav")
            if not wav_path.exists():
                continue

            with label_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            drone = data.get("drone", {})
            source = drone.get("sound_source")
            if source is None:
                continue
            if self.source_filter is not None and source not in self.source_filter:
                continue

            with wave.open(str(wav_path), "rb") as wav:
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                num_frames = wav.getnframes()

            recordings.append(
                RecordingMeta(
                    recording_id=label_path.parent.name,
                    wav_path=wav_path,
                    label_path=label_path,
                    source=str(source),
                    distance_m=self._parse_float(drone.get("distance")),
                    height_m=self._parse_float(drone.get("height")),
                    azimuth_deg=self._parse_float(drone.get("azimuth")),
                    rotation=self._parse_rotation(drone.get("rotation")),
                    sample_rate=sample_rate,
                    channels=channels,
                    num_frames=num_frames,
                )
            )
        return recordings

    def _build_windows(self) -> list[WindowMeta]:
        windows: list[WindowMeta] = []
        for recording in self.recordings:
            split_start, split_end = self._split_bounds(recording.num_frames)
            split_frames = split_end - split_start
            if split_frames <= 0:
                continue

            if self.window_seconds is None:
                windows.append(
                    WindowMeta(
                        recording=recording,
                        start_frame=split_start,
                        num_frames=split_frames,
                    )
                )
                continue

            window_frames = max(1, int(round(self.window_seconds * recording.sample_rate)))
            if self.hop_seconds is None:
                hop_frames = window_frames
            else:
                hop_frames = max(1, int(round(self.hop_seconds * recording.sample_rate)))

            if split_frames < window_frames:
                if self.pad_short_windows:
                    windows.append(
                        WindowMeta(
                            recording=recording,
                            start_frame=split_start,
                            num_frames=window_frames,
                        )
                    )
                continue

            start_frame = split_start
            last_start = split_end - window_frames
            while start_frame <= last_start:
                windows.append(
                    WindowMeta(
                        recording=recording,
                        start_frame=start_frame,
                        num_frames=window_frames,
                    )
                )
                start_frame += hop_frames
        return windows

    def _split_bounds(self, total_frames: int) -> tuple[int, int]:
        split_frame = int(total_frames * self.split_ratio)
        if self.split == "train":
            return 0, split_frame
        if self.split == "test":
            return split_frame, total_frames
        return 0, total_frames

    @staticmethod
    def _parse_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value))
        except ValueError:
            return None

    @staticmethod
    def _parse_rotation(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _index_or_default(mapping: dict[object, int], key: object) -> int:
        return mapping.get(key, -1)

    def _sorted_unique(self, field: str) -> tuple[float, ...]:
        values = {
            getattr(recording, field)
            for recording in self.recordings
            if getattr(recording, field) is not None
        }
        return tuple(sorted(values))

    def _rotation_order(self) -> tuple[str, ...]:
        preferred = ("Front", "Right", "Back", "Left")
        available = {
            recording.rotation for recording in self.recordings if recording.rotation is not None
        }
        ordered = [item for item in preferred if item in available]
        ordered.extend(sorted(available - set(ordered)))
        return tuple(ordered)

    @staticmethod
    def _read_audio_window(
        wav_path: Path,
        start_frame: int,
        num_frames: int,
        expected_channels: int,
        pad_short: bool,
    ) -> np.ndarray:
        with wave.open(str(wav_path), "rb") as wav:
            n_frames = wav.getnframes()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            if channels != expected_channels:
                raise RuntimeError(
                    f"Expected {expected_channels} channels, found {channels} in {wav_path}"
                )

            wav.setpos(min(start_frame, n_frames))
            frame_bytes = wav.readframes(num_frames)

        if sample_width == 4:
            audio = np.frombuffer(frame_bytes, dtype="<i4")
            scale = float(2**31)
            audio = audio.astype(np.float32) / scale
        elif sample_width == 2:
            audio = np.frombuffer(frame_bytes, dtype="<i2")
            scale = float(2**15)
            audio = audio.astype(np.float32) / scale
        elif sample_width == 1:
            audio = np.frombuffer(frame_bytes, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        else:
            raise RuntimeError(f"Unsupported sample width {sample_width} in {wav_path}")

        if audio.size % expected_channels != 0:
            audio = audio[: audio.size - (audio.size % expected_channels)]
        audio = audio.reshape(-1, expected_channels)

        if pad_short and audio.shape[0] < num_frames:
            pad = np.zeros((num_frames - audio.shape[0], expected_channels), dtype=np.float32)
            audio = np.vstack([audio, pad])
        return audio


def create_dataloader(
    dataset: AcousticUAVDataset,
    batch_size: int = 8,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader with train split shuffling by default."""
    if shuffle is None:
        shuffle = dataset.split == "train"

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
