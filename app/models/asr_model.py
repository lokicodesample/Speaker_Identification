from transformers import pipeline
from config import Config

class ASRModel:
    def __init__(self):
        self.model = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device="cpu",
            torch_dtype="auto",
            generate_kwargs={"language": "en"}
        )

    def transcribe(self, audio_path):
        result = self.model(
            audio_path,
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=False
        )
        return result['text']