import os

def generate_timestamps(labels, group_size, hop_size, sample_rate):
    """Convert labels into (speaker, start_time, end_time)"""
    segments = []
    segment_start = 0
    current_speaker = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current_speaker:
            start_time = segment_start * group_size * hop_size / sample_rate
            end_time = i * group_size * hop_size / sample_rate
            segments.append((current_speaker, round(start_time, 2), round(end_time, 2)))
            segment_start = i
            current_speaker = labels[i]

    # Handle last segment
    start_time = segment_start * group_size * hop_size / sample_rate
    end_time = len(labels) * group_size * hop_size / sample_rate
    segments.append((current_speaker, round(start_time, 2), round(end_time, 2)))

    return segments

def save_diarization_output(segments, output_path="results/diarization.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for speaker, start, end in segments:
            line = f"Speaker {speaker} [{start:.2f}s - {end:.2f}s]"
            print(line)
            f.write(line + '\n')

def merge_short_segments(segments, min_duration=0.8):
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev_speaker, prev_start, prev_end = merged[-1]
        cur_speaker, cur_start, cur_end = seg
        if cur_speaker == prev_speaker and (cur_end - cur_start) < min_duration:
            merged[-1] = (prev_speaker, prev_start, cur_end)
        else:
            merged.append(seg)
    return merged
