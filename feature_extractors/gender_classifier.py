import librosa
import numpy as np
from .feature_extractor import FeatureExtractor

from .gender_recognition.utils import create_model

class GenderClassifier(FeatureExtractor):
    def __init__(self, tag_type, model, config):
        super().__init__(tag_type, model, config)
        self.tag_type = tag_type
        self.config = config

        # construct the model
        self.model = create_model()
        # load the saved/trained weights
        self.model.load_weights(model)

    def extract_intermediate_features(self, file_name, config):
        """
        Extract feature from audio file `file_name`
            Features supported:
                - MFCC (mfcc)
                - Chroma (chroma)
                - MEL Spectrogram Frequency (mel)
                - Contrast (contrast)
                - Tonnetz (tonnetz)
            e.g:
            `features = extract_feature(path, mel=True, mfcc=True)`
        """
        mfcc = config.get("mfcc")
        chroma = config.get("chroma")
        mel = config.get("mel")
        contrast = config.get("contrast")
        tonnetz = config.get("tonnetz")
        X, sample_rate = librosa.core.load(file_name)
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
        return result

    def extract_features(self, audio_path):
        audio_features = self.extract_intermediate_features(audio_path, self.config).reshape(1, -1)
        gender_probabilities = self.model.predict(audio_features)
        self.features["probabilities"] = gender_probabilities
        male_prob = gender_probabilities[0][0]
        if male_prob < 0.30 or male_prob > 0.70:
            female_prob = 1 - male_prob
            gender = "male" if male_prob > female_prob else "female"
        else:
            gender = "inconclusive"

        return gender

    def get_tag_type(self):
        return self.tag_type