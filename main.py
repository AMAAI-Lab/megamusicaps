from feature_extractors.feature_extractor import FeatureExtractor, FeatureExtractors
from feature_extractors.essentia_extractors import EssentiaFeatureExtractor, EssentiaVoiceExtractor
from feature_extractors.beatnet_extractor import BeatNetExtractor
from feature_extractors.btc_chord_extractor import BTCChordExtractor
from feature_extractors.key_classification_extractor import KeyClassificationExtractor

import os

from feature_extractors.gender_classifier import GenderClassifier

from caption_generator import CaptionGenerator

import argparse
import yaml
import json

import warnings

import time
from tqdm import tqdm

class MusicCaptioner:
	def __init__(self, config_file_path):
		self.config_file_path = config_file_path

		self.configs = self.load_configs(self.config_file_path)

		self.enable_caption_generation = self.configs["pipeline"]["enable_caption_generation"]

		self.input_file_path = self.configs["files"]["input"]
		self.output_file_path = self.configs["files"]["output"]
		self.source_separated_file_dir = self.configs["paths"]["source_separated_audio"]
		self.saved_features_dir = self.configs["paths"]["saved_features"]

		self.temp_file_path = self.configs["paths"]["temp"]

		self.active_extractors = []
		self.feature_extractors = []

		if self.configs["extractors"]["mood_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.MOOD_EXTRACTOR.value, EssentiaFeatureExtractor("mood", self.configs["extractors"]["mood_extractor"]["model"], self.configs["extractors"]["mood_extractor"], self.configs["extractors"]["mood_extractor"]["model_metadata"], self.configs["extractors"]["mood_extractor"]["embedding_model"], 5, 0.1))
			if ("source" in self.configs["extractors"]["mood_extractor"].keys()):
				self.feature_extractors[FeatureExtractors.MOOD_EXTRACTOR.value].set_source(self.configs["extractors"]["mood_extractor"]["source"])
			self.active_extractors.append(FeatureExtractors.MOOD_EXTRACTOR.value)
		else:
			self.feature_extractors.insert(FeatureExtractors.MOOD_EXTRACTOR.value, None)

		if self.configs["extractors"]["genre_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.GENRE_EXTRACTOR.value, EssentiaFeatureExtractor("genre", self.configs["extractors"]["genre_extractor"]["model"], self.configs["extractors"]["genre_extractor"], self.configs["extractors"]["genre_extractor"]["model_metadata"], self.configs["extractors"]["genre_extractor"]["embedding_model"], 4, 0.1))
			if ("source" in self.configs["extractors"]["genre_extractor"].keys()):
				self.feature_extractors[FeatureExtractors.GENRE_EXTRACTOR.value].set_source(self.configs["extractors"]["genre_extractor"]["source"])
			self.active_extractors.append(FeatureExtractors.GENRE_EXTRACTOR.value)
		else:
			self.feature_extractors.insert(FeatureExtractors.GENRE_EXTRACTOR.value, None)

		if self.configs["extractors"]["instrument_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.INSTRUMENT_EXTRACTOR.value, EssentiaFeatureExtractor("instrument", self.configs["extractors"]["instrument_extractor"]["model"], self.configs["extractors"]["instrument_extractor"], self.configs["extractors"]["instrument_extractor"]["model_metadata"], self.configs["extractors"]["instrument_extractor"]["embedding_model"], 7, 0.1))
			if ("source" in self.configs["extractors"]["instrument_extractor"].keys()):
				self.feature_extractors[FeatureExtractors.INSTRUMENT_EXTRACTOR.value].set_source(self.configs["extractors"]["instrument_extractor"]["source"])
			self.active_extractors.append(FeatureExtractors.INSTRUMENT_EXTRACTOR.value)
		else:
			self.feature_extractors.insert(FeatureExtractors.INSTRUMENT_EXTRACTOR.value, None)

		if self.configs["extractors"]["auto_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.AUTO_EXTRACTOR.value, EssentiaFeatureExtractor("autotags", self.configs["extractors"]["auto_extractor"]["model"], self.configs["extractors"]["auto_extractor"], self.configs["extractors"]["auto_extractor"]["model_metadata"], self.configs["extractors"]["auto_extractor"]["embedding_model"], 8, 0.1))
			if ("source" in self.configs["extractors"]["auto_extractor"].keys()):
				self.feature_extractors[FeatureExtractors.AUTO_EXTRACTOR.value].set_source(self.configs["extractors"]["auto_extractor"]["source"])
			self.active_extractors.append(FeatureExtractors.AUTO_EXTRACTOR.value)
		else:
			self.feature_extractors.insert(FeatureExtractors.AUTO_EXTRACTOR.value, None)

		if self.configs["extractors"]["voice_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.VOICE_EXTRACTOR.value, EssentiaVoiceExtractor("voice", self.configs["extractors"]["voice_extractor"]["model"], self.configs["extractors"]["voice_extractor"], self.configs["extractors"]["voice_extractor"]["model_metadata"], self.configs["extractors"]["voice_extractor"]["embedding_model"]))
			if ("source" in self.configs["extractors"]["voice_extractor"].keys()):
				self.feature_extractors[FeatureExtractors.VOICE_EXTRACTOR.value].set_source(self.configs["extractors"]["voice_extractor"]["source"])
			self.active_extractors.append(FeatureExtractors.VOICE_EXTRACTOR.value)
		else:
			self.feature_extractors.insert(FeatureExtractors.VOICE_EXTRACTOR.value, None)

		if self.configs["extractors"]["gender_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.GENDER_EXTRACTOR.value, EssentiaVoiceExtractor("gender", self.configs["extractors"]["gender_extractor"]["model"], self.configs["extractors"]["gender_extractor"], self.configs["extractors"]["gender_extractor"]["model_metadata"], self.configs["extractors"]["gender_extractor"]["embedding_model"]))
			if ("source" in self.configs["extractors"]["gender_extractor"].keys()):
				self.feature_extractors[FeatureExtractors.GENDER_EXTRACTOR.value].set_source(self.configs["extractors"]["gender_extractor"]["source"])
			self.active_extractors.append(FeatureExtractors.GENDER_EXTRACTOR.value)
		else:
			self.feature_extractors.insert(FeatureExtractors.GENDER_EXTRACTOR.value, None)

		if self.configs["extractors"]["beatnet_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.BEATNET_EXTRACTOR.value, BeatNetExtractor("beats", self.configs["extractors"]["beatnet_extractor"]["model"], self.configs["extractors"]["beatnet_extractor"]))
			if ("source" in self.configs["extractors"]["beatnet_extractor"].keys()):
				self.feature_extractors[FeatureExtractors.BEATNET_EXTRACTOR.value].set_source(self.configs["extractors"]["beatnet_extractor"]["source"])
			self.active_extractors.append(FeatureExtractors.BEATNET_EXTRACTOR.value)
		else:
			self.feature_extractors.insert(FeatureExtractors.BEATNET_EXTRACTOR.value, None)

		if self.configs["extractors"]["btc_chord_extractor"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.BTC_CHORD_EXTRACTOR.value, BTCChordExtractor("chords", self.configs["extractors"]["btc_chord_extractor"]["model"], self.configs["extractors"]["btc_chord_extractor"], self.configs["extractors"]["btc_chord_extractor"]["config_file"]))
			if ("source" in self.configs["extractors"]["btc_chord_extractor"].keys()):
				self.feature_extractors[FeatureExtractors.BTC_CHORD_EXTRACTOR.value].set_source(self.configs["extractors"]["btc_chord_extractor"]["source"])
			self.active_extractors.append(FeatureExtractors.BTC_CHORD_EXTRACTOR.value)
		else:
			self.feature_extractors.insert(FeatureExtractors.BTC_CHORD_EXTRACTOR.value, None)

		if self.configs["extractors"]["gender_classifier"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.GENDER_CLASSIFIER.value, GenderClassifier("gender", self.configs["extractors"]["gender_classifier"]["model"], self.configs["extractors"]["gender_classifier"]))
			if ("source" in self.configs["extractors"]["gender_classifier"].keys()):
				self.feature_extractors[FeatureExtractors.GENDER_CLASSIFIER.value].set_source(self.configs["extractors"]["gender_classifier"]["source"])
			self.active_extractors.append(FeatureExtractors.GENDER_CLASSIFIER.value)
		else:
			self.feature_extractors.insert(FeatureExtractors.GENDER_CLASSIFIER.value, None)

		if self.configs["extractors"]["key_classifier"]["active"] :
			self.feature_extractors.insert(FeatureExtractors.KEY_CLASSIFIER.value, KeyClassificationExtractor("key", self.configs["extractors"]["key_classifier"]["model"], self.configs["extractors"]["key_classifier"]))
			if ("source" in self.configs["extractors"]["key_classifier"].keys()):
				self.feature_extractors[FeatureExtractors.KEY_CLASSIFIER.value].set_source(self.configs["extractors"]["key_classifier"]["source"])
			self.active_extractors.append(FeatureExtractors.KEY_CLASSIFIER.value)
		else:
			self.feature_extractors.insert(FeatureExtractors.KEY_CLASSIFIER.value, None)

		self.caption_generator = CaptionGenerator(self.configs["caption_generator"]["api_key"], self.configs["caption_generator"]["model_id"])

	def load_configs(self, file_path):
		configs = {}
		with open(file_path, 'r') as f:
			configs = yaml.safe_load(f)
		return configs

	def get_audio_paths(self, file_path):
		audio_paths = []
		with open(file_path, 'r') as f:
			for row in f:
				audio_path=json.loads(row)
				audio_paths.append(audio_path["location"])
		return audio_paths

	def caption_audio(self, snippet_path):
		audio_tags = {}
		for extractor in self.active_extractors:
			if not self.feature_extractors[extractor].get_source() == "raw":
				source_splitted_path = self.source_separated_file_dir + "/" + os.path.splitext(os.path.basename(snippet_path))[0] + "/" + self.feature_extractors[extractor].get_source() + ".mp3"
				if not os.path.exists(source_splitted_path):
					source_splitted_path = snippet_path
			else:
				source_splitted_path = snippet_path

			feature_tags = self.feature_extractors[extractor].extract_features(source_splitted_path)
			if self.feature_extractors[extractor].get_config_value("save_features", False):
				features_dir = self.saved_features_dir + "/" + os.path.splitext(os.path.basename(snippet_path))[0] + "/"
				if not os.path.exists(features_dir):
					os.makedirs(features_dir)
				self.feature_extractors[extractor].save_extracted_features(features_dir)
			audio_tags[self.feature_extractors[extractor].get_tag_type()] = feature_tags

		if (self.enable_caption_generation):
			prompt = self.caption_generator.create_prompt(audio_tags)
			caption = self.caption_generator.generate_caption(prompt)
			audio_tags["caption"] = caption
		audio_tags["location"] = snippet_path
		return audio_tags

	def caption_all(self):
		audio_paths = self.get_audio_paths(self.input_file_path)
		audio_tags = []
		for audio_path in tqdm(audio_paths, desc="Captioning Progress", unit="audio"):
			try:
				caption = self.caption_audio(audio_path)
			except:
				print("Error captioning : ", audio_path)
				caption = {"location": audio_path, "caption": "!!!Error"}
			audio_tags.append(caption)
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
