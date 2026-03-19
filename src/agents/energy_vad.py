"""
Lightweight energy-based VAD fallback for realtime user speech detection.
This avoids heavy model inference when Silero runs too slowly on CPU.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
from livekit import rtc
from livekit.agents import vad as lk_vad
from livekit.agents.utils import log_exceptions

logger = logging.getLogger(__name__)


@dataclass
class EnergyVADOptions:
    threshold: float
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float


class EnergyVAD(lk_vad.VAD):
    """Simple energy-threshold VAD for low-latency speech detection."""

    def __init__(
        self,
        *,
        threshold: float = 0.012,
        min_speech_duration: float = 0.15,
        min_silence_duration: float = 0.45,
        prefix_padding_duration: float = 0.15,
        update_interval: float = 0.02,
    ) -> None:
        super().__init__(
            capabilities=lk_vad.VADCapabilities(update_interval=update_interval)
        )
        self._opts = EnergyVADOptions(
            threshold=threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
        )

    def stream(self) -> "EnergyVADStream":
        return EnergyVADStream(self, self._opts)


class EnergyVADStream(lk_vad.VADStream):
    def __init__(self, vad: EnergyVAD, opts: EnergyVADOptions) -> None:
        super().__init__(vad)
        self._opts = opts
        self._input_sample_rate = 0
        self._samples_index = 0
        self._timestamp = 0.0

        self._prefix_frames: Deque[Tuple[rtc.AudioFrame, float]] = deque()
        self._prefix_duration = 0.0

        self._pending_frames: List[rtc.AudioFrame] = []
        self._pending_duration = 0.0

        self._speech_frames: List[rtc.AudioFrame] = []
        self._speech_duration = 0.0
        self._silence_duration = 0.0
        self._speaking = False

    def _push_prefix(self, frame: rtc.AudioFrame, duration: float) -> None:
        if self._opts.prefix_padding_duration <= 0:
            return
        self._prefix_frames.append((frame, duration))
        self._prefix_duration += duration
        while self._prefix_duration > self._opts.prefix_padding_duration:
            old_frame, old_duration = self._prefix_frames.popleft()
            self._prefix_duration -= old_duration

    def _frame_energy(self, frame: rtc.AudioFrame) -> float:
        data = frame.data
        if not isinstance(data, np.ndarray):
            data = np.frombuffer(data, dtype=np.int16)
        if data.size == 0:
            return 0.0
        rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))
        return float(rms / 32768.0)

    def _reset_buffers(self) -> None:
        self._pending_frames.clear()
        self._pending_duration = 0.0
        self._speech_frames.clear()
        self._speech_duration = 0.0
        self._silence_duration = 0.0
        self._speaking = False

    def _emit_event(
        self,
        *,
        event_type: lk_vad.VADEventType,
        frames: List[rtc.AudioFrame],
        probability: float,
        inference_duration: float,
    ) -> None:
        self._event_ch.send_nowait(
            lk_vad.VADEvent(
                type=event_type,
                samples_index=self._samples_index,
                timestamp=self._timestamp,
                speech_duration=self._speech_duration,
                silence_duration=self._silence_duration,
                frames=frames,
                probability=probability,
                inference_duration=inference_duration,
            )
        )

    @log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        async for item in self._input_ch:
            if isinstance(item, lk_vad.VADStream._FlushSentinel):
                if self._speaking and self._speech_frames:
                    self._emit_event(
                        event_type=lk_vad.VADEventType.END_OF_SPEECH,
                        frames=list(self._speech_frames),
                        probability=1.0,
                        inference_duration=0.0,
                    )
                self._reset_buffers()
                continue

            if not isinstance(item, rtc.AudioFrame):
                continue

            frame = item
            if not self._input_sample_rate:
                self._input_sample_rate = frame.sample_rate
            if frame.sample_rate != self._input_sample_rate:
                continue

            frame_duration = (
                frame.samples_per_channel / frame.sample_rate
                if frame.sample_rate
                else 0.0
            )
            self._samples_index += frame.samples_per_channel
            self._timestamp += frame_duration

            start_time = time.perf_counter()
            energy = self._frame_energy(frame)
            inference_duration = time.perf_counter() - start_time

            self._emit_event(
                event_type=lk_vad.VADEventType.INFERENCE_DONE,
                frames=[frame],
                probability=min(1.0, energy / max(self._opts.threshold, 1e-6)),
                inference_duration=inference_duration,
            )

            self._push_prefix(frame, frame_duration)

            if energy >= self._opts.threshold:
                self._pending_frames.append(frame)
                self._pending_duration += frame_duration
                self._silence_duration = 0.0

                if self._speaking:
                    self._speech_frames.append(frame)
                    self._speech_duration += frame_duration
                    continue

                if self._pending_duration >= self._opts.min_speech_duration:
                    self._speaking = True
                    prefix_frames = [f for f, _ in self._prefix_frames]
                    self._speech_frames = prefix_frames + self._pending_frames
                    self._speech_duration = self._pending_duration
                    self._emit_event(
                        event_type=lk_vad.VADEventType.START_OF_SPEECH,
                        frames=list(self._speech_frames),
                        probability=1.0,
                        inference_duration=inference_duration,
                    )
                    self._pending_frames = []
                    self._pending_duration = 0.0
                continue

            # energy below threshold
            if self._speaking:
                self._silence_duration += frame_duration
                self._speech_frames.append(frame)
                if self._silence_duration >= self._opts.min_silence_duration:
                    self._emit_event(
                        event_type=lk_vad.VADEventType.END_OF_SPEECH,
                        frames=list(self._speech_frames),
                        probability=1.0,
                        inference_duration=inference_duration,
                    )
                    self._reset_buffers()
            else:
                self._pending_frames.clear()
                self._pending_duration = 0.0
