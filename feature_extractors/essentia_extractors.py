from .feature_extractor import FeatureExtractor
from essentia.standard import MonoLoader
from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import json
import numpy as np

class EssentiaFeatureExtractor(FeatureExtractor):
	def __init__(self, tag_type, model_path, config, model_metadata_path, embedding_model, max_num_tags = 5, tag_threshold = 0.1):
		self.tag_type = tag_type
		self.config = config
		self.model_path = model_path
		self.model_metadata_path = model_metadata_path
		self.model, self.model_metadata = self.load_model(model_path, model_metadata_path)

		self.embedding_model = embedding_model
		self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=self.embedding_model, output="PartitionedCall:1")

		self.max_num_tags = max_num_tags
		self.tag_threshold = tag_threshold

		self.features = {}

	def load_model(self, model_path, model_metadata_path):
		with open(model_metadata_path, 'r') as json_file:
			metadata = json.load(json_file)
		model = TensorflowPredict2D(graphFilename=model_path)
		return model, metadata

	def set_max_num_tags(self, max_num_tags):
		self.max_num_tags = max_num_tags

	def set_tag_threshold(self, tag_threshold):
		self.tag_threshold = tag_threshold

	def load_audio(self, audio_path):
		audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
		return audio

	def extract_features(self, snippet_path):
		audio_features = self.load_audio(snippet_path)
		embeddings = self.embedding_model(audio_features)
		tags, cs = self.get_tags(embeddings)
		if len(tags) > 0:
			return tags[0]
		else: 
			return ''

	def get_tags(self, embeddings):
		predictions = self.model(embeddings)
		mean_act=np.mean(predictions,0)
		self.features["mean_act"] = mean_act

		ind = np.argpartition(mean_act, -self.max_num_tags)[-self.max_num_tags:]

		tags=[]
		confidence_score=[]
		for i in ind:
			print(self.model_metadata['classes'][i] + str(mean_act[i]))
			if mean_act[i]>self.tag_threshold:
				tags.append(self.model_metadata['classes'][i])
				confidence_score.append(mean_act[i])

		ind=np.argsort(-np.array(confidence_score))
		tags = [tags[i] for i in ind]
		confidence_score=np.round((np.array(confidence_score)[ind]).tolist(),4).tolist()

		return tags, confidence_score

	def get_tag_type(self):
		return self.tag_type


class EssentiaVoiceExtractor(EssentiaFeatureExtractor):
	def load_model(self, model_path, model_metadata_path):
		with open(model_metadata_path, 'r') as json_file:
			metadata = json.load(json_file)
		model = TensorflowPredict2D(graphFilename=model_path, output="model/Softmax")
		return model, metadata

	def get_tags(self, embeddings):
		predictions = self.model(embeddings)
		mean_act=np.mean(predictions,0)
		self.features["mean_act"] = mean_act

		ind = np.argmax(mean_act)
		tag=self.model_metadata['classes'][ind]

		return [tag], mean_act.tolist()