import os
import csv
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from datetime import datetime
import logging

matplotlib.use('Agg')  # Set backend before other matplotlib imports

# Initialize logger
logger = logging.getLogger(__name__)


class Visualization:
    def __init__(self, results, audio_path):
        self.results = results
        self.audio_path = audio_path

        # Validate audio file exists
        if not os.path.exists(self.audio_path):
            logger.error(f"Audio file not found: {self.audio_path}")
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

        # Configure visualization directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.visual_dir = os.path.join(base_dir, 'static', 'visualizations')
        os.makedirs(self.visual_dir, exist_ok=True)
        logger.info(f"Visualization directory initialized: {self.visual_dir}")

    def generate_visualizations(self):
        """Generate all visualizations and return filenames"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return {
                'timeline': self._generate_timeline_plot(timestamp),
                'spectrogram': self._generate_spectrogram(timestamp)
            }
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}", exc_info=True)
            raise

    def _generate_timeline_plot(self, timestamp):
        """Generate speaker timeline visualization"""
        filename = f"timeline_{timestamp}.png"
        filepath = os.path.join(self.visual_dir, filename)

        try:
            plt.figure(figsize=(16, 6))
            speakers = list({s['speaker'] for s in self.results})
            colormap = plt.colormaps.get_cmap('tab20')  # Updated for matplotlib 3.7+
            colors = colormap(np.linspace(0, 1, len(speakers)))

            for idx, seg in enumerate(self.results):
                color_idx = speakers.index(seg['speaker'])
                plt.plot(
                    [seg['start'], seg['end']],
                    [idx, idx],
                    color=colors[color_idx],
                    linewidth=8,
                    solid_capstyle='round'
                )

            plt.xlabel('Time (seconds)', fontsize=12)
            plt.title('Speaker Activity Timeline', fontsize=14)
            plt.yticks([])
            plt.grid(True, alpha=0.2)
            plt.tight_layout()

            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            logger.info(f"Timeline visualization saved: {filepath}")
            return filename

        except Exception as e:
            plt.close()  # Ensure figure is closed on error
            logger.error(f"Failed to generate timeline: {str(e)}", exc_info=True)
            raise

    def _generate_spectrogram(self, timestamp):
        """Generate audio spectrogram visualization"""
        filename = f"spectrogram_{timestamp}.png"
        filepath = os.path.join(self.visual_dir, filename)

        try:
            y, sr = librosa.load(self.audio_path, sr=None)  # Keep original sample rate
            plt.figure(figsize=(12, 6))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Audio Frequency Spectrogram')
            plt.tight_layout()

            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            logger.info(f"Spectrogram saved: {filepath}")
            return filename

        except Exception as e:
            plt.close()
            logger.error(f"Failed to generate spectrogram: {str(e)}", exc_info=True)
            raise

    def generate_csv_report(self, timestamp):
        """Generate CSV transcript report"""
        filename = f"report_{timestamp}.csv"
        filepath = os.path.join(self.visual_dir, filename)

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Speaker', 'Start (s)', 'End (s)', 'Duration (s)', 'Text'])
                for seg in self.results:
                    writer.writerow([
                        seg['speaker'],
                        f"{seg['start']:.2f}",
                        f"{seg['end']:.2f}",
                        f"{seg['duration']:.2f}",
                        seg['text']
                    ])
            logger.info(f"CSV report generated successfully: {filepath}")
            return filename
        except Exception as e:
            logger.error(f"CSV report generation failed: {str(e)}", exc_info=True)
            raise