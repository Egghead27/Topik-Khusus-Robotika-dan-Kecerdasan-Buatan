# tts.py

import wave
import subprocess
from pathlib import Path
from typing import List, Union

from piper.voice import PiperVoice


class TextToSpeech:
    """
    Piper TTS wrapper.

    - Combines multiple OCR chunks
    - Saves one audio file
    - Replaces previous output
    - Plays generated audio
    """

    def __init__(
        self,
        model_path: str,
        output_path: str = "output/audio.wav",
        auto_play: bool = True,
    ):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.auto_play = auto_play

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        print("Loading Piper model...")
        self.voice = PiperVoice.load(
            self.model_path
        )
        print("Piper loaded.")


    # ========================================================
    # Public API
    # ========================================================

    def synthesize(
        self,
        text: Union[str, List[str]]
    ) -> Path:
        """
        Generate speech.

        Args:
            text:
                str
                or list of OCR chunks

        Returns:
            output wav path
        """

        # remove previous image result
        self._clear_previous()


        if isinstance(text, list):
            text = self._combine_chunks(text)


        text = text.strip()


        if not text:
            raise ValueError(
                "Empty TTS input"
            )


        self._save_wav(
            text,
            self.output_path
        )


        if self.auto_play:
            self.play()


        return self.output_path


    # ========================================================
    # Piper
    # ========================================================

    def _save_wav(
        self,
        text: str,
        path: Path
    ):

        chunks = iter(
            self.voice.synthesize(text)
        )

        try:
            first = next(chunks)

        except StopIteration:
            raise RuntimeError(
                "Piper generated no audio"
            )


        with wave.open(
            str(path),
            "wb"
        ) as wav:

            wav.setnchannels(
                first.sample_channels
            )

            wav.setsampwidth(
                first.sample_width
            )

            wav.setframerate(
                first.sample_rate
            )


            wav.writeframes(
                first.audio_int16_bytes
            )


            for chunk in chunks:
                wav.writeframes(
                    chunk.audio_int16_bytes
                )


    # ========================================================
    # Playback
    # ========================================================

    def play(self):

        subprocess.Popen(
            [
                "powershell",
                "-c",
                (
                    "(New-Object Media.SoundPlayer "
                    f"'{self.output_path}').PlaySync();"
                )
            ]
        )


    # ========================================================
    # Utils
    # ========================================================

    def _combine_chunks(
        self,
        chunks: List[str]
    ) -> str:

        return " ".join(
            chunk.strip()
            for chunk in chunks
            if chunk.strip()
        )


    def _clear_previous(self):

        if self.output_path.exists():
            self.output_path.unlink()