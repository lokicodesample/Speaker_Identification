import os
from pydub import AudioSegment
from app.models.asr_model import ASRModel
from app.models.diarization_model import DiarizationModel
from app.utils.file_handling import convert_to_wav
import logging


class AudioProcessor:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.wav_path = None
        self.results = []
        self.speaker_count = 0
        self.duration = 0
        self.asr_model = ASRModel()
        self.diarization_model = DiarizationModel()

    def process_audio(self):
        """Main audio processing pipeline with error handling"""
        try:
            # Convert to WAV format first
            self.wav_path = convert_to_wav(self.audio_path)
            if not os.path.exists(self.wav_path):
                raise RuntimeError("Failed to convert audio to WAV format")

            # Load audio and calculate duration
            audio = AudioSegment.from_wav(self.wav_path)
            self.duration = len(audio) / 1000  # Convert to seconds

            # Perform speaker diarization
            diarization = self.diarization_model.process(self.wav_path)

            # Process each speech segment
            for segment in diarization.itertracks(yield_label=True):
                turn, _, speaker = segment
                start = round(turn.start, 2)
                end = round(turn.end, 2)
                duration = round(end - start, 2)

                # Extract audio chunk
                chunk = audio[int(start * 1000):int(end * 1000)]
                chunk_path = f"{self.audio_path}_chunk_{start}_{end}.wav"
                chunk.export(chunk_path, format="wav")

                # Transcribe audio chunk
                transcription = self.asr_model.transcribe(chunk_path)
                os.remove(chunk_path)  # Clean up temporary chunk

                self.results.append({
                    'speaker': speaker,
                    'start': start,
                    'end': end,
                    'duration': duration,
                    'text': transcription
                })

            self.speaker_count = len({r['speaker'] for r in self.results})

            return {
                'results': self.results,
                'speaker_count': self.speaker_count,
                'duration': self.duration,
                'original_filename': os.path.basename(self.audio_path)
            }

        except Exception as e:
            # Clean up any temporary files
            if hasattr(self, 'wav_path') and self.wav_path and os.path.exists(self.wav_path):
                os.remove(self.wav_path)
            logging.error(f"Audio processing error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Audio processing failed: {str(e)}")