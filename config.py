import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'dev-key-123')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'static', 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg'}
    HF_TOKEN = os.environ.get('HF_TOKEN')
    AUDIO_SAMPLE_RATE = 16000
    COLOR_PALETTE = 'tab20'

    @staticmethod
    def init_app(app):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        visualizations_dir = os.path.join(app.config['UPLOAD_FOLDER'], '..', 'visualizations')
        os.makedirs(visualizations_dir, exist_ok=True)