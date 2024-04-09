from .feature_extractor import FeatureExtractor

from nnAudio import Spectrogram
import librosa
import torch
from torch import nn
import numpy as np
import argparse
import json

from .noisy_student_extraction.models import HPCPModel, NoisyHPCPModel, SingleModel


class NoisyStudentExtractor(FeatureExtractor):
	def __init__(self, tag_type, model_path, config, model_metadata_path, max_num_tags = 5, tag_threshold = 0.1):
		self.tag_type = tag_type
		self.config = config
		self.model_path = model_path
		self.model_metadata_path = model_metadata_path
		
		self.max_num_tags = max_num_tags
		self.tag_threshold = tag_threshold

		self.features = {}

	

	def get_tag_type(self):
		return self.tag_type
	

	def extract_features(self, audio_path):
		with open(self.model_metadata_path, "r") as f:
			moodClasses = [line.rstrip() for line in f]

		melspec_op = Spectrogram.MelSpectrogram(sr=44100).cuda()

		audio, sr = librosa.load(audio_path, sr=44100)
		audio = torch.tensor(audio).cuda()
		melspec = melspec_op(audio)
		melspec = melspec.unsqueeze(0)
		melspec = nn.AvgPool2d((1, 10), stride=(1, 10))(melspec)
		melspec = melspec.cpu().detach().numpy().squeeze()[:, :10000]
		assert melspec.shape[0] == 128
		assert melspec.shape[1] > 0
		print(melspec.shape)

		model = SingleModel(128, 256, 56).cuda()
		model.eval()
		S = torch.load(self.model_path, map_location=torch.device("cuda"))
		model.load_state_dict(S)

		melspec = melspec.T
		melspec = np.log1p(melspec)
		melspec = melspec / 9.6
		melspec = torch.tensor(melspec)


		new_melspec = []
		for i in range(0, melspec.shape[0], 80):
			if melspec[i:i+80, :].shape[0] == 80:
				new_melspec.append(melspec[i:i+80, :])
		new_melspec = torch.stack(new_melspec, dim=0)

		new_melspec = torch.transpose(new_melspec, 1, 2)
		new_melspec = new_melspec.unsqueeze(1).cuda()
		print(new_melspec.shape)

		with torch.no_grad():
			y_raw = model(new_melspec)
			y = torch.mean(y_raw, dim=0)
			self.features["Raw_prediction"] = y.cpu().numpy()
			print("Output:")
			print(y.cpu().numpy())

			sorted, indices = torch.sort(y, descending=True)
			print("Corresonding indices:")
			print(indices)

			num_tags_above_threshold = torch.count_nonzero(y > self.tag_threshold)
			predictedClasses = indices[0 : min(self.max_num_tags, num_tags_above_threshold)]

			tags = [moodClasses[classIndex] for classIndex in predictedClasses]
			
		return tags