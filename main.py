from feature_extractors.feature_extractor import FeatureExtractor, FeatureExtractors
from essentia.standard import MonoLoader
from feature_extractors.essentia_extractors import EssentiaFeatureExtractor, EssentiaVoiceExtractor
# from feature_extractors.effects_extractor import EffectsExtractor
from caption_generator import CaptionGenerator

import argparse
import yaml
import json

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="essentia")

class MusicCaptioner:
	def __init__(self, config_file_path):
		self.config_file_path = config_file_path

		self.configs = self.load_configs(self.config_file_path)

		self.input_file_path = self.configs["files"]["input"]
		self.output_file_path = self.configs["files"]["output"]

		self.active_extractors = []
		self.feature_extractors = []

		if self.configs["extractors"]["mood_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.MOOD_EXTRACTOR.value, EssentiaFeatureExtractor("mood", self.configs["extractors"]["mood_extractor"]["model"], self.configs["extractors"]["mood_extractor"]["model_metadata"], self.configs["extractors"]["mood_extractor"]["embedding_model"], 5, 0.1))
			self.active_extractors.append(FeatureExtractors.MOOD_EXTRACTOR.value)

		if self.configs["extractors"]["genre_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.GENRE_EXTRACTOR.value, EssentiaFeatureExtractor("genre", self.configs["extractors"]["genre_extractor"]["model"], self.configs["extractors"]["genre_extractor"]["model_metadata"], self.configs["extractors"]["genre_extractor"]["embedding_model"], 4, 0.1))
			self.active_extractors.append(FeatureExtractors.GENRE_EXTRACTOR.value)

		if self.configs["extractors"]["instrument_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.INSTRUMENT_EXTRACTOR.value, EssentiaFeatureExtractor("instrument", self.configs["extractors"]["instrument_extractor"]["model"], self.configs["extractors"]["instrument_extractor"]["model_metadata"], self.configs["extractors"]["instrument_extractor"]["embedding_model"], 7, 0.1))
			self.active_extractors.append(FeatureExtractors.INSTRUMENT_EXTRACTOR.value)

		if self.configs["extractors"]["auto_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.AUTO_EXTRACTOR.value, EssentiaFeatureExtractor("autotags", self.configs["extractors"]["auto_extractor"]["model"], self.configs["extractors"]["auto_extractor"]["model_metadata"], self.configs["extractors"]["auto_extractor"]["embedding_model"], 8, 0.1))
			self.active_extractors.append(FeatureExtractors.AUTO_EXTRACTOR.value)

		if self.configs["extractors"]["voice_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.VOICE_EXTRACTOR.value, EssentiaVoiceExtractor("voice", self.configs["extractors"]["voice_extractor"]["model"], self.configs["extractors"]["voice_extractor"]["model_metadata"], self.configs["extractors"]["voice_extractor"]["embedding_model"]))
			self.active_extractors.append(FeatureExtractors.VOICE_EXTRACTOR.value)

		if self.configs["extractors"]["gender_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.GENDER_EXTRACTOR.value, EssentiaVoiceExtractor("gender", self.configs["extractors"]["gender_extractor"]["model"], self.configs["extractors"]["gender_extractor"]["model_metadata"], self.configs["extractors"]["gender_extractor"]["embedding_model"]))
			self.active_extractors.append(FeatureExtractors.GENDER_EXTRACTOR.value)

		self.caption_generator = CaptionGenerator(self.configs["caption_generator"]["api_key"], self.configs["caption_generator"]["model_id"])

	def load_configs(self, file_path):
		configs = {}
		with open(file_path, 'r') as f:
			configs = yaml.safe_load(f)
		return configs

	def load_audio(self, audio_path):
		audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
		return audio

	def get_audio_paths(self, file_path):
		audio_paths = []
		with open(file_path, 'r') as f:
			for row in f:
				audio_path=json.loads(row)
				audio_paths.append(audio_path["location"])
		return audio_paths

	def caption_audio(self, snippet_path):
		audio_features = self.load_audio(snippet_path)
		audio_tags = {}
		for extractor in self.active_extractors:
			feature_tags, feature_cs = self.feature_extractors[extractor].extract_features(audio_features)
			if len(feature_tags) > 0:
				audio_tags[self.feature_extractors[extractor].get_tag_type()] = feature_tags[0]

		print("TAGS ", audio_tags)
		prompt = self.caption_generator.create_prompt(audio_tags)
		print("PROMPT ", prompt)
		caption = self.caption_generator.generate_caption(prompt)
		print("CAPTION ", caption)
		audio_tags["caption"] = caption
		audio_tags["location"] = snippet_path
		return audio_tags

	def caption_all(self):
		audio_paths = self.get_audio_paths(self.input_file_path)
		audio_tags = []
		for audio_path in audio_paths:
			audio_tags.append(self.caption_audio(audio_path))
		return audio_tags

	def save_captions(self, audio_tags):
		with open(self.output_file_path, 'w') as out_f:
			for tags in audio_tags:
				out_f.write(json.dumps(tags) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for audio snippets.")
    parser.add_argument("config_file", help="Path to the file with configs for generation of captions")
    args = parser.parse_args()

    music_captioner = MusicCaptioner(args.config_file)
    captions = music_captioner.caption_all()
    music_captioner.save_captions(captions)
