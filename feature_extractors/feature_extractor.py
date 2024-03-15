from enum import Enum

class FeatureExtractors(Enum):
	MOOD_EXTRACTOR = 0
	GENRE_EXTRACTOR = 1
	INSTRUMENT_EXTRACTOR = 2
	AUTO_EXTRACTOR = 3
	VOICE_EXTRACTOR = 4
	GENDER_EXTRACTOR = 5
	BEATNET_EXTRACTOR = 6
	BTC_CHORD_EXTRACTOR = 7
	GENDER_CLASSIFIER = 8
	KEY_CLASSIFIER = 9

# Base class for feature extraction
class FeatureExtractor:
	def __init__(self, model):
		self.features = {}
		self.model = model
		self.source = "raw"

	def get_tag_type(self):
		raise NotImplementedError("Subclasses must implement this method")

	def extract_features(self):
		raise NotImplementedError("Subclasses must implement this method")

	def set_source(self, source):
		if source not in ["raw", "vocals", "drums", "bass", "other"]:
			self.source = "raw"
		else:
			self.source = source

	def get_source(self):
		return self.source