import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Path to your dataset
dataset_path = "dataset"

# Emotion label map (assuming folder names)
emotions = ['happy', 'sad', 'angry', 'neutral']

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print("Error:", e)
        return None

# Collect features and labels
X, y = [], []
for emotion in emotions:
    folder_path = os.path.join(dataset_path, emotion)
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            path = os.path.join(folder_path, filename)
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(emotion)

X = np.array(X)
y = np.array(y)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "emotion_model.pkl")
print("✅ Model trained and saved as 'emotion_model.pkl'")
