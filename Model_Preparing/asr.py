import whisper
import numpy as np
# Whisper expects float32 between -1 and 1
import tempfile
import os
import soundfile as sf

model = whisper.load_model("base")  # or "tiny", "small", "medium", "large"

def transcribe_segment(audio_slice, sample_rate):
    """Transcribe audio segment using Whisper"""


    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        sf.write(temp.name, audio_slice, sample_rate)
        result = model.transcribe(temp.name)
        return result['text']


def save_segment_audio(audio, sample_rate, speaker, index, start, end, out_folder="results/speakers/"):
    os.makedirs(out_folder, exist_ok=True)
    segment = audio[int(start * sample_rate):int(end * sample_rate)]
    filename = f"{out_folder}/speaker{speaker}_seg{index}.wav"
    sf.write(filename, segment, sample_rate)
