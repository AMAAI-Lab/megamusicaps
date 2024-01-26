from .feature_extractor import FeatureExtractor
import numpy as np
from scipy.signal import find_peaks, correlate

from BeatNet.BeatNet import BeatNet

class BeatNetExtractor(FeatureExtractor):
	def __init__(self, tag_type, model, config):
		self.tag_type = tag_type
		self.model = model
		self.mode = config["mode"]
		self.plot = config["plot"]
		self.thread = config["thread"]

		self.estimator = BeatNet(1, mode=self.mode, inference_model=self.model, plot=self.plot, thread=self.thread)

	def extract_features(self, audio_path):
		beats = self.get_beats(audio_path)
		bpm, repeating_pattern, inflection_points = self.identify_pattern(beats)

		tags = dict()
		tags["bpm"] = bpm
		tags["beat_pattern"] = list(repeating_pattern)

		# tags = [bpm, repeating_pattern, inflection_points]
		return tags

	def get_beats(self, audio_path):
		return self.estimator.process(audio_path)

	def identify_pattern(self, beats):
		# Extract timestamps and beats
		timestamps, beat_values = beats[:, 0], beats[:, 1]

		# Calculate time differences between consecutive beats
		time_diff = np.diff(timestamps)

		# Calculate BPM (beats per minute)
		from scipy import stats
		time_diff_pdf = np.zeros((149, 2))
		time_diff_pdf[:, 0] = np.arange(1, 150) / 100
		# Correlation between time differene data and a Gaussian distribution curve
		for position in np.arange(1, 150):
			corr_result = np.sum(stats.norm.pdf(time_diff, position/100, 0.02))
			time_diff_pdf[position-1, 1] = corr_result
		# Find the peak
		bpm = 60 / time_diff_pdf[np.argmax(time_diff_pdf[:, 1]), 0]

		# Find inflection points where BPM changes
		inflection_points, _ = find_peaks(-time_diff, height=0)

		# Extract the beat pattern using autocorrelation
		autocorr = correlate(beat_values, beat_values, mode='full')
		autocorr = autocorr[len(autocorr)//2:]

		# Find peaks in autocorrelation to identify repeating pattern
		pattern_indices, _ = find_peaks(autocorr, height=0)

		repeating_pattern = beat_values[:pattern_indices[0]]

		return bpm, repeating_pattern, inflection_points

	def get_tag_type(self):
		return self.tag_type