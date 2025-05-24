import base64
import json
from typing import Optional
from pydantic import BaseModel
from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    KeypadEntry,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class ExotelFrameSerializer(FrameSerializer):
    """
    Serializer for Exotel voice platform WebSocket protocol.

    This serializer handles converting between Pipecat frames and Exotel's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and proper
    handling of Exotel's expected audio format.

    Exotel expects 16-bit, 8 kHz, mono PCM (little-endian) audio data, base64-encoded
    in the WebSocket payload. This serializer ensures that audio frames are resampled
    and encoded as required, and that incoming audio is converted to the pipeline's
    sample rate as needed.

    Attributes:
        _stream_sid: The Exotel stream session identifier.
        _params: Configuration parameters for the serializer.
        _exotel_sample_rate: Sample rate used by Exotel (fixed at 8kHz).
        _sample_rate: Input sample rate for the pipeline.
        _resampler: Audio resampler for format conversion.
    """

    class InputParams(BaseModel):
        """
        Configuration parameters for ExotelFrameSerializer.

        Attributes:
            exotel_sample_rate: Sample rate used by Exotel, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
        """
        exotel_sample_rate: int = 8000  # Exotel's fixed rate (8kHz)
        sample_rate: Optional[int] = None  # Pipeline input rate

    def __init__(self, stream_sid: str, params: Optional['ExotelFrameSerializer.InputParams'] = None):
        """
        Initialize the ExotelFrameSerializer.

        Args:
            stream_sid: The Exotel stream session identifier.
            params: Configuration parameters (optional).
        """
        self._stream_sid = stream_sid
        self._params = params or ExotelFrameSerializer.InputParams()
        self._exotel_sample_rate = self._params.exotel_sample_rate
        self._sample_rate = 0  # Pipeline input rate
        self._resampler = create_default_resampler()

    @property
    def type(self) -> FrameSerializerType:
        """
        Gets the serializer type.

        Returns:
            The serializer type, either TEXT or BINARY.
        """
        return FrameSerializerType.TEXT

    async def setup(self, frame: StartFrame):
        """
        Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """
        Serializes a Pipecat frame to Exotel WebSocket format.

        Handles conversion of various frame types to Exotel WebSocket messages.
        For audio frames, ensures audio is resampled to 8kHz PCM and base64-encoded.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.
        """
        if isinstance(frame, StartInterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer, ensure_ascii=False)

        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # If the frame's sample rate differs from Exotel's expected rate (8kHz),
            # resample the audio to 8kHz PCM
            if frame.sample_rate != self._exotel_sample_rate:
                # Resample to Exotel's 8kHz PCM format
                serialized_data = await self._resampler.resample(
                    data, frame.sample_rate, self._exotel_sample_rate
                )
            else:
                # No conversion needed if already at 8kHz
                serialized_data = data

            # Encode the PCM data to base64 for transmission
            payload = base64.b64encode(serialized_data).decode("utf-8")

            # Format the response according to Exotel's expectations
            answer = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            }
            return json.dumps(answer, ensure_ascii=False)

        elif isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            # Pass through transport messages as JSON
            return json.dumps(frame.message, ensure_ascii=False)

        # Return None for unhandled frames
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """
        Deserializes Exotel WebSocket data to Pipecat frames.

        Handles conversion of Exotel media events to appropriate Pipecat frames.
        For audio, decodes base64 PCM and resamples to the pipeline's sample rate.

        Args:
            data: The raw WebSocket data from Exotel.

        Returns:
            A Pipecat frame corresponding to the Exotel event, or None if unhandled.
        """
        message = json.loads(data)

        if message["event"] == "media":
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            # If the pipeline input rate differs from Exotel's 8kHz,
            # resample the incoming audio
            if self._sample_rate != self._exotel_sample_rate:
                # Resample from Exotel's 8kHz PCM to pipeline's rate
                deserialized_data = await self._resampler.resample(
                    payload, self._exotel_sample_rate, self._sample_rate
                )
            else:
                # No conversion needed if pipeline also uses 8kHz
                deserialized_data = payload

            # Create an audio frame with the PCM data
            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
            )
            return audio_frame

        elif message["event"] == "dtmf":
            digit = message.get("dtmf", {}).get("digit")
            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                # Handle case where string doesn't match any enum value
                return None

        # Return None for unhandled events
        return None
