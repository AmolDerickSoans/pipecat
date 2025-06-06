#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements Tavus as a sink transport layer"""

import asyncio
import time
from typing import Optional

import aiohttp
from daily.daily import AudioData, VideoFrame
from loguru import logger

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from pipecat.transports.services.tavus import TavusCallbacks, TavusParams, TavusTransportClient


class TavusVideoService(AIService):
    """
    Service class that proxies audio to Tavus and receives both audio and video in return.

    It uses the `TavusTransportClient` to manage the session and handle communication. When
    audio is sent, Tavus responds with both audio and video streams, which are then routed
    through Pipecat’s media pipeline.

    In use cases such as with `DailyTransport`, this results in two distinct virtual rooms:
        - **Tavus room**: Contains the Tavus Avatar and the Pipecat Bot.
        - **User room**: Contains the Pipecat Bot and the user.

    Args:
        api_key (str): Tavus API key used for authentication.
        replica_id (str): ID of the Tavus voice replica to use for speech synthesis.
        persona_id (str): ID of the Tavus persona. Defaults to "pipecat0" to use the Pipecat TTS voice.
        session (aiohttp.ClientSession): Async HTTP session used for communication with Tavus.
        **kwargs: Additional arguments passed to the parent `AIService` class.
    """

    def __init__(
        self,
        *,
        api_key: str,
        replica_id: str,
        persona_id: str = "pipecat0",  # Use `pipecat0` so that your TTS voice is used in place of the Tavus persona
        session: aiohttp.ClientSession,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key
        self._session = session
        self._replica_id = replica_id
        self._persona_id = persona_id

        self._other_participant_has_joined = False
        self._client: Optional[TavusTransportClient] = None

        self._conversation_id: str
        self._resampler = create_default_resampler()

        self._audio_buffer = bytearray()
        self._queue = asyncio.Queue()
        self._send_task: Optional[asyncio.Task] = None

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        callbacks = TavusCallbacks(
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
        )
        self._client = TavusTransportClient(
            bot_name="Pipecat",
            callbacks=callbacks,
            api_key=self._api_key,
            replica_id=self._replica_id,
            persona_id=self._persona_id,
            session=self._session,
            params=TavusParams(
                audio_in_enabled=True,
                video_in_enabled=True,
            ),
        )
        await self._client.setup(setup)

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()
        self._client = None

    async def _on_participant_left(self, participant, reason):
        participant_id = participant["id"]
        logger.info(f"Participant left {participant_id}, reason: {reason}")

    async def _on_participant_joined(self, participant):
        participant_id = participant["id"]
        logger.info(f"Participant joined {participant_id}")
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._client.capture_participant_video(
                participant_id, self._on_participant_video_frame, 30
            )
            await self._client.capture_participant_audio(
                participant_id=participant_id,
                callback=self._on_participant_audio_data,
                sample_rate=self._client.out_sample_rate,
            )

    async def _on_participant_video_frame(
        self, participant_id: str, video_frame: VideoFrame, video_source: str
    ):
        frame = OutputImageRawFrame(
            image=video_frame.buffer,
            size=(video_frame.width, video_frame.height),
            format=video_frame.color_format,
        )
        frame.transport_source = video_source
        await self.push_frame(frame)

    async def _on_participant_audio_data(
        self, participant_id: str, audio: AudioData, audio_source: str
    ):
        frame = OutputAudioRawFrame(
            audio=audio.audio_frames,
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
        )
        frame.transport_source = audio_source
        await self.push_frame(frame)

    def can_generate_metrics(self) -> bool:
        return True

    async def get_persona_name(self) -> str:
        return await self._client.get_persona_name()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.start(frame)
        await self._create_send_task()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._end_conversation()
        await self._cancel_send_task()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._end_conversation()
        await self._cancel_send_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._handle_interruptions()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStartedFrame):
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()
            self._current_idx_str = str(frame.id)
        elif isinstance(frame, TTSAudioRawFrame):
            await self._queue_audio(frame.audio, frame.sample_rate, done=False)
        elif isinstance(frame, TTSStoppedFrame):
            await self._queue_audio(b"\x00\x00", self._client.in_sample_rate, done=True)
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)

    async def _handle_interruptions(self):
        await self._cancel_send_task()
        await self._create_send_task()
        await self._client.send_interrupt_message()

    async def _end_conversation(self):
        await self._client.stop()
        self._other_participant_has_joined = False

    async def _queue_audio(self, audio: bytes, in_rate: int, done: bool):
        await self._queue.put((audio, in_rate, done))

    async def _create_send_task(self):
        if not self._send_task:
            self._queue = asyncio.Queue()
            self._send_task = self.create_task(self._send_task_handler())

    async def _cancel_send_task(self):
        if self._send_task:
            await self.cancel_task(self._send_task)
            self._send_task = None

    # TODO (Filipi): this should be all that is needed use this Microphone Echo mode
    # https://docs.tavus.io/sections/conversational-video-interface/layers-and-modes-overview#microphone-echo
    # This would allow us to send an audio stream for the replica to repeat
    # Checking with Tavus what is the right way to create the Persona to make it work
    # async def _send_task_handler(self):
    #    while True:
    #        (audio, in_rate, done) = await self._queue.get()
    #        await self._client.write_raw_audio_frames(audio)

    async def _send_task_handler(self):
        # Daily app-messages have a 4kb limit and also a rate limit of 20
        # messages per second. Below, we only consider the rate limit because 1
        # second of a 24000 sample rate would be 48000 bytes (16-bit samples and
        # 1 channel). So, that is 48000 / 20 = 2400, which is below the 4kb
        # limit (even including base64 encoding). For a sample rate of 16000,
        # that would be 32000 / 20 = 1600.
        sample_rate = self._client.out_sample_rate
        MAX_CHUNK_SIZE = int((sample_rate * 2) / 20)

        audio_buffer = bytearray()
        samples_sent = 0
        start_time = time.time()

        while True:
            (audio, in_rate, done) = await self._queue.get()

            if done:
                # Send any remaining audio.
                if len(audio_buffer) > 0:
                    await self._client.encode_audio_and_send(
                        bytes(audio_buffer), done, self._current_idx_str
                    )
                await self._client.encode_audio_and_send(audio, done, self._current_idx_str)
                audio_buffer.clear()
            else:
                audio = await self._resampler.resample(audio, in_rate, sample_rate)
                audio_buffer.extend(audio)
                while len(audio_buffer) >= MAX_CHUNK_SIZE:
                    chunk = audio_buffer[:MAX_CHUNK_SIZE]
                    audio_buffer = audio_buffer[MAX_CHUNK_SIZE:]

                    # Compute wait time for synchronization
                    wait = start_time + (samples_sent / sample_rate) - time.time()
                    if wait > 0:
                        await asyncio.sleep(wait)

                    await self._client.encode_audio_and_send(
                        bytes(chunk), done, self._current_idx_str
                    )

                    # Update timestamp based on number of samples sent
                    samples_sent += len(chunk) // 2  # 2 bytes per sample (16-bit)
