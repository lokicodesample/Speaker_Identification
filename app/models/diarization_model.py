from pyannote.audio import Pipeline
from config import Config

class DiarizationModel:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=Config.HF_TOKEN
        )

    def process(self, audio_path):
        return self.pipeline(audio_path)