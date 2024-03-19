from .feature_extractor import FeatureExtractor
import numpy as np
from scipy.signal import correlate
import torch
from .btc_chord_extraction.btc_model import *
from .btc_chord_extraction.utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths
from .btc_chord_extraction.utils import logger
import argparse
import warnings
import os

warnings.filterwarnings('ignore')
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class BTCChordExtractor(FeatureExtractor):
	def __init__(self, tag_type, model, config, config_file = "./btc_chord_extraction/run_config.yaml", audio_dir='./btc_chord_extraction/test', save_dir='./btc_chord_extraction/test', voca=False):
		super().__init__(tag_type, model, config)
		self.tag_type = tag_type
		self.audio_dir = audio_dir
		self.save_dir = save_dir
		self.voca = voca
		self.config = config
		self.module_config = HParams.load(config_file)
		self._configure_model(model)

	def _configure_model(self, model):
		if self.voca:
			self.module_config.feature['large_voca'] = True
			self.module_config.model['num_chords'] = 170
			model_file = f'{self.save_dir}/btc_model_large_voca.pt'
			idx_to_chord = idx2voca_chord()
			logger.info("label type: large voca")
		else:
			model_file = f'{self.save_dir}/btc_model.pt'
			self.idx_to_chord = idx2chord
			logger.info("label type: Major and minor")

		self.model = BTC_model(config=self.module_config.model).to(device)

		# Load model
		if os.path.isfile(model):
			checkpoint = torch.load(model,  map_location=torch.device('cpu'))
			self.mean = checkpoint['mean']
			self.std = checkpoint['std']
			self.model.load_state_dict(checkpoint['model'])
			logger.info("restore model")

	def extract_features(self, audio_path):
		# logger.info("Processing audio file: %s" % audio_path)
		feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, self.module_config)
		# logger.info("Audio file loaded and feature computation success : %s" % audio_path)

		feature = feature.T
		feature = (feature - self.mean) / self.std
		time_unit = feature_per_second
		n_timestep = self.module_config.model['timestep']

		num_pad = n_timestep - (feature.shape[0] % n_timestep)
		feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
		num_instance = feature.shape[0] // n_timestep

		start_time = 0.0
		lines = []
		with torch.no_grad():
			self.model.eval()
			feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
			self.features["prediction"] = np.array([])
			for t in range(num_instance):
				self_attn_output, _ = self.model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
				prediction, _ = self.model.output_layer(self_attn_output)
				prediction = prediction.squeeze()
				self.features["prediction"] = np.concatenate([self.features["prediction"], prediction.cpu()])
				for i in range(n_timestep):
					if t == 0 and i == 0:
						prev_chord = prediction[i].item()
						continue
					if prediction[i].item() != prev_chord:
						lines.append('%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), self.idx_to_chord[prev_chord]))
						start_time = time_unit * (n_timestep * t + i)
						prev_chord = prediction[i].item()
					if t == num_instance - 1 and i + num_pad == n_timestep:
						if start_time != time_unit * (n_timestep * t + i):
							lines.append('%.3f %.3f %s\n' % (
								start_time, time_unit * (n_timestep * t + i), self.idx_to_chord[prev_chord]))
						break

		return lines


	def get_tag_type(self):
		return self.tag_type