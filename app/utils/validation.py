from werkzeug.utils import secure_filename
from config import Config


def validate_audio_file(file):
    """Validate uploaded audio file"""
    if not file:
        return False, "No file selected"

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return False, "Invalid file type"

    return True, ""