import os
import ffmpeg
import time
from config import Config
from werkzeug.utils import secure_filename


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def convert_to_wav(input_path):
    """Convert any audio file to WAV format"""
    try:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_converted.wav"

        if os.path.exists(output_path):
            os.remove(output_path)

        (
            ffmpeg.input(input_path)
            .output(output_path,
                    ar=Config.AUDIO_SAMPLE_RATE,
                    ac=1,
                    loglevel="error")
            .run(capture_stdout=True, capture_stderr=True)
        )
        return output_path
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode('utf-8')}")
    except Exception as e:
        raise RuntimeError(f"Conversion error: {str(e)}")


def cleanup_files(*paths, max_age_hours=24):
    """Clean up files older than specified hours"""
    now = time.time()
    cutoff = now - (max_age_hours * 3600)

    for path in paths:
        if path and os.path.exists(path):
            try:
                file_age = os.path.getmtime(path)
                if file_age < cutoff:
                    os.remove(path)
            except Exception:
                pass