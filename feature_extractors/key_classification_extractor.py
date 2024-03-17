from .feature_extractor import FeatureExtractor

import jams
import librosa
from .key_classification.classifier import KeyClassifier
from .key_classification.feature import read_features

class KeyClassificationExtractor(FeatureExtractor):
	def __init__(self, tag_type, model, config):
		super().__init__(tag_type, model, config)
		self.tag_type = tag_type
		self.model = model
		self.config = config
		self.classifier = KeyClassifier(self.model)

	def get_tag_type(self):
		return self.tag_type

	def extract_features(self, audio_path):
		result = self.get_key(audio_path)
		self.features = self.classifier.get_features()
		return result

	def get_source(self):
		return self.source

	def get_key(self, input_file):
		features = read_features(input_file)
		tonic, mode = self.classifier.estimate_key(features)
		if mode == 'major':
			result = tonic
		elif mode == 'minor':
			result = '{}m'.format(tonic)
		else:
			result = '{}{}'.format(tonic, mode)
		return result