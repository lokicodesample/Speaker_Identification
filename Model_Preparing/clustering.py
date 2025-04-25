from sklearn.cluster import KMeans
import numpy as np

def average_mfccs(mfccs, group_size=5):
    """Average every N frames to reduce noise"""
    grouped = []
    for i in range(0, len(mfccs), group_size):
        chunk = mfccs[i:i+group_size]
        if len(chunk) == group_size:
            grouped.append(np.mean(chunk, axis=0))
    return np.array(grouped)

def cluster_embeddings(embeddings, num_speakers=2):
    """Cluster MFCC embeddings into groups (speakers)"""
    model = KMeans(n_clusters=num_speakers, random_state=0)
    labels = model.fit_predict(embeddings)
    return labels
