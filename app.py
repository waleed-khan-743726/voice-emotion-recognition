from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import librosa

app = Flask(__name__)
model = joblib.load('emotion_model.pkl')

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['audio']
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            features = extract_features(file_path)
            if features is not None and len(features) == 40:
                prediction = model.predict([features])
                result = prediction[0]
            else:
                result = "Feature extraction failed."

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
