import logging
logging.basicConfig(level=logging.DEBUG)
import os
import matplotlib

matplotlib.use('Agg')  # Must be before other imports
import torchvision
import warnings
import speechbrain
from flask import Flask
from config import Config

# Suppress warnings
torchvision.disable_beta_transforms_warning()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
speechbrain.utils.parameter_transfer.WARN_SYMLINKS = False


def create_app(config_class=Config):
    # Get absolute path to static folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_path = os.path.join(base_dir, 'static')

    # Ensure required directories exist
    required_dirs = {
        'uploads': os.path.join(static_path, 'uploads'),
        'visualizations': os.path.join(static_path, 'visualizations'),
        'images': os.path.join(static_path, 'images')
    }

    for dir_name, dir_path in required_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Create Flask app with explicit static configuration
    app = Flask(
        __name__,
        static_folder=static_path,
        static_url_path='/static'
    )

    # Configure app
    app.config.from_object(config_class)
    config_class.init_app(app)

    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    # Verify configuration
    print(f"Static folder configured at: {app.static_folder}")
    print(f"Uploads path: {required_dirs['uploads']}")
    print(f"Visualizations path: {required_dirs['visualizations']}")

    return app