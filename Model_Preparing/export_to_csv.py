import csv

def export_segments_to_csv(segments, texts, out_path="results/diarization_transcript.csv"):
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Speaker", "Start Time", "End Time", "Transcript"])
        for ((speaker, start, end), text) in zip(segments, texts):
            writer.writerow([f"Speaker {speaker}", start, end, text])
    print(f"✅ CSV exported to: {out_path}")
