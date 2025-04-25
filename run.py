from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    # Ensure upload and visualization directories exist
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'static', 'uploads')
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'static', 'visualizations')
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    app.run(host='0.0.0.0', port=5000, debug=True)




