from enum import Enum
import numpy as np
import torch
import os

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
	def __init__(self, tag_type, model, config):
		self.features = {}
		self.model = model
		self.config = config
		self.source = "raw"
		self.tag_type = tag_type

	def get_tag_type(self):
		raise NotImplementedError("Subclasses must implement this method")

	def extract_features(self):
		raise NotImplementedError("Subclasses must implement this method")

	def save_extracted_features(self, _dir):
		_format = self.get_config_value("format", "npy")
		if not os.path.isdir(_dir):
			print("[FeatureExtractor][save_extracted_features] Directory doesn't exist " + str(_dir))
			return False
		for feature in self.features.keys():
			if _format == "npy":
				path = _dir + "/" + str(self.tag_type) + "_" + str(feature) + ".npy"
				if isinstance(self.features[feature], np.ndarray):
					np.save(path, self.features[feature])
					return True
				else:
					np.save(path, self.features[feature].numpy())
					return True
			elif _format == "pt":
				path = _dir + "/" + str(self.tag_type) + "_" + str(feature) + ".pt"
				torch.save(self.features[feature], path)
				return True
			elif _format == "h5py":
				path = _dir + "/" + str(self.tag_type) + "_" + str(feature) + ".h5"
				with h5py.File(path, 'w') as hf:
				    hf.create_dataset(str(feature) + "_dataset", data=tensor.numpy())
				return True
			else:
				print("[FeatureExtractor][save_extracted_features] Invalid format " + str(_format))
				return False

	def set_source(self, source):
		if source not in ["raw", "vocals", "drums", "bass", "other"]:
			self.source = "raw"
		else:
			self.source = source

	def get_config_value(self, key, default):
		if key in self.config.keys():
			return self.config[key]
		else:
			return default

	def get_source(self):
		return self.source