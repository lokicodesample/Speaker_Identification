from flask import Blueprint, render_template, request, send_from_directory, flash, redirect, url_for, abort, current_app
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import logging
import time
from app.audio_processor import AudioProcessor
from app.visualization import Visualization
from app.utils.file_handling import allowed_file, cleanup_files
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)


@main_bp.before_app_request
def cleanup_old_files():
    """Clean up files older than 24 hours"""
    try:
        # Use current_app's static_folder reference
        uploads_dir = os.path.join(current_app.static_folder, 'uploads')
        visualizations_dir = os.path.join(current_app.static_folder, 'visualizations')

        cleanup_files(uploads_dir)
        cleanup_files(visualizations_dir)

        logger.info(f"Cleanup completed. Uploads: {len(os.listdir(uploads_dir))} files, "
                    f"Visualizations: {len(os.listdir(visualizations_dir))} files")
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}", exc_info=True)


@main_bp.route('/debug/files')
def debug_files():
    """Debug endpoint to check generated files"""
    try:
        files = {
            'uploads': glob.glob(os.path.join(current_app.static_folder, 'uploads', '*')),
            'visualizations': glob.glob(os.path.join(current_app.static_folder, 'visualizations', '*'))
        }
        return {
            'status': 'success',
            'uploads_dir': os.path.join(current_app.static_folder, 'uploads'),
            'visualizations_dir': os.path.join(current_app.static_folder, 'visualizations'),
            'files': {k: [os.path.basename(f) for f in v] for k, v in files.items()}
        }
    except Exception as e:
        logger.error(f"Debug error: {str(e)}", exc_info=True)
        return {'status': 'error', 'message': str(e)}, 500


@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/analyze', methods=['POST'])
def analyze():
    if 'audio_file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('main.index'))

    file = request.files['audio_file']
    if not file or file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('main.index'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Allowed: .wav, .mp3, .m4a', 'error')
        return redirect(url_for('main.index'))

    save_path = None
    try:
        # Save with timestamp prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = secure_filename(file.filename)
        save_path = os.path.join(
            current_app.static_folder,
            'uploads',
            f"{timestamp}_{original_filename}"
        )
        file.save(save_path)
        logger.info(f"File saved to: {save_path}")

        # Process audio
        processor = AudioProcessor(save_path)
        results = processor.process_audio()
        logger.info(f"Processing completed. Found {results['speaker_count']} speakers")

        # Generate visualizations
        visualizer = Visualization(results['results'], save_path)
        visualizations = visualizer.generate_visualizations()
        time.sleep(1)  # Ensure files are fully written

        # Generate CSV report
        csv_report = visualizer.generate_csv_report(timestamp)
        time.sleep(1)

        # Verify files exist before rendering
        for vis_type, filename in visualizations.items():
            full_path = os.path.join(current_app.static_folder, 'visualizations', filename)
            if not os.path.exists(full_path):
                raise RuntimeError(f"Generated file not found: {full_path}")

        return render_template('results.html',
                               results=results['results'],
                               speaker_count=results['speaker_count'],
                               duration=results['duration'],
                               visualizations=visualizations,
                               csv_report=csv_report,
                               original_filename=f"{timestamp}_{original_filename}")

    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
                logger.info(f"Cleaned up file: {save_path}")
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {str(cleanup_error)}")
        return render_template('error.html',
                               message=f"Processing failed: {str(e)}"), 500


@main_bp.route('/download/<path:filename>')
def download(filename):
    """Secure file download handler"""
    try:
        safe_filename = secure_filename(filename)
        logger.info(f"Download requested: {safe_filename}")

        # Split into directory and filename
        if '/' in safe_filename:
            folder, clean_filename = safe_filename.split('/', 1)
            if folder not in ['uploads', 'visualizations']:
                abort(400, "Invalid directory in filename")
        else:
            folder = 'uploads'
            clean_filename = safe_filename

        file_path = os.path.join(current_app.static_folder, folder, clean_filename)

        logger.info(f"Looking for file at: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            abort(404, description="File not found. It may have been deleted or never generated.")

        logger.info(f"Serving file from: {file_path}")
        return send_from_directory(
            directory=os.path.join(current_app.static_folder, folder),
            path=clean_filename,
            as_attachment=True
        )

    except ValueError:
        logger.error("Invalid filename format")
        abort(400, "Invalid filename format")
    except Exception as e:
        logger.error(f"Download error: {str(e)}", exc_info=True)
        abort(500, description=f"Download failed: {str(e)}")