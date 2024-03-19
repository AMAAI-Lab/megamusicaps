from .feature_extractor import FeatureExtractor
import numpy as np
from scipy.signal import find_peaks, correlate

from .beatnet_extraction.BeatNet import BeatNet

class BeatNetExtractor(FeatureExtractor):
	def __init__(self, tag_type, model, config):
		self.tag_type = tag_type
		self.model = model
		self.config = config
		self.mode = config["mode"]
		self.plot = config["plot"]
		self.thread = config["thread"]

		self.estimator = BeatNet(1, mode=self.mode, inference_model=self.model, plot=self.plot, thread=self.thread)

	def extract_features(self, audio_path):
		beats = self.get_beats(audio_path)
		self.features = self.estimator.get_features()
		bpm, repeating_pattern, inflection_points = self.identify_pattern(beats)

		tags = dict()
		tags["bpm"] = bpm
		tags["beat_pattern"] = list(repeating_pattern)

		# tags = [bpm, repeating_pattern, inflection_points]
		return tags

	def get_beats(self, audio_path):
		return self.estimator.process(audio_path)

	def identify_pattern(self, beats):

		if len(beats) == 0:
				# Handle the case where beats array is empty
				return [], [], []

		# Extract timestamps and beats
		timestamps, beat_values = beats[:, 0], beats[:, 1]

		# Calculate time differences between consecutive beats
		time_diff = np.diff(timestamps)

		# Calculate BPM (beats per minute)
		bpm = 60 / np.mean(time_diff)

		# Find inflection points where BPM changes
		inflection_points, _ = find_peaks(-time_diff, height=0)

		# Extract the beat pattern using autocorrelation
		autocorr = correlate(beat_values, beat_values, mode='full')
		autocorr = autocorr[len(autocorr)//2:]

		# Find peaks in autocorrelation to identify repeating pattern
		pattern_indices, _ = find_peaks(autocorr, height=0)
		if len(pattern_indices) > 0:
			repeating_pattern = beat_values[:pattern_indices[0]]
		else:
			repeating_pattern = []

		return bpm, repeating_pattern, inflection_points

	def get_tag_type(self):
		return self.tag_type