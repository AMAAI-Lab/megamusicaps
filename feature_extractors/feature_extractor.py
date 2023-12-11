from enum import Enum

class FeatureExtractors(Enum):
	MOOD_EXTRACTOR = 0
	GENRE_EXTRACTOR = 1
	INSTRUMENT_EXTRACTOR = 2
	AUTO_EXTRACTOR = 3
	VOICE_EXTRACTOR = 4
	GENDER_EXTRACTOR = 4

# Base class for feature extraction
class FeatureExtractor:
	def __init__(self, model):
		self.features = {}
		self.model = model

	def get_tag_type(self):
		raise NotImplementedError("Subclasses must implement this method")

	def extract_features(self):
		raise NotImplementedError("Subclasses must implement this method")