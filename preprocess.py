from feature_extractors.source_separation.source_separator import separate_audio
import argparse
import yaml
import json
from pydub import AudioSegment
import os

class Preprocessor:
	def __init__(self, config_file_path):
		self.config_file_path = config_file_path

		self.configs = self.load_configs(self.config_file_path)

		self.input_file_path = self.configs["files"]["raw_input"]
		self.processed_file_path = self.configs["files"]["input"]
		self.audio_paths = self.get_audio_paths(self.input_file_path)
		self.source_separated_file_dir = self.configs["paths"]["source_separated_audio"]
		self.split_files_dir = self.configs["paths"]["split_audio"]

	def get_audio_paths(self, file_path):
		audio_paths = []
		with open(file_path, 'r') as f:
			for row in f:
				audio_path=json.loads(row)
				audio_paths.append(audio_path["location"])
		return audio_paths

	def load_configs(self, file_path):
		configs = {}
		with open(file_path, 'r') as f:
			configs = yaml.safe_load(f)
		return configs

	def split_mp3(self, input_file_path, output_dir, segment_duration=30):

		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		
		# Load the audio file
		audio = AudioSegment.from_file(input_file_path, format="mp3")

		# Get the base file name without extension
		base_file_name = os.path.splitext(os.path.basename(input_file_path))[0]

		# Calculate the number of segments
		num_segments = len(audio) // (segment_duration * 1000)

		output_files = []

		# Split the audio into 30-second segments
		for i in range(num_segments):
			start_time = i * segment_duration * 1000
			end_time = (i + 1) * segment_duration * 1000
			segment = audio[start_time:end_time]

			# Export each segment to a new file with the original file name and index
			output_file_path = f"{output_dir}/{base_file_name}_{i + 1}.mp3"
			segment.export(output_file_path, format="mp3")
			output_files.append(output_file_path)

		return output_files

	def separate_source(self,input_file_path, output_dir):
		# Get the base file name without extension
		base_file_name = os.path.splitext(os.path.basename(input_file_path))[0]
		output_dir = f"{output_dir}/{base_file_name}"
		separated_audio_paths = separate_audio(input_file_path, output_dir)

		return separated_audio_paths

	def save_files(self, audio_paths):
		data = [{"location": file_path} for file_path in audio_paths]

		with open(self.processed_file_path, 'w') as out_f:
			for _data in data:
				out_f.write(json.dumps(_data) + '\n')

	def preprocess_all(self):
		audio_paths = self.get_audio_paths(self.input_file_path)
		audio_tags = []
		silence_removed_paths = []
		split_audio_paths = []
		source_separated_audio_paths = []
		for audio_path in audio_paths:
			split_files = self.split_mp3(audio_path, self.split_files_dir, 30)
			split_audio_paths += split_files

		for split_audio_file in split_audio_paths:
			audio_tags.append(self.separate_source(split_audio_file, self.source_separated_file_dir))

		self.save_files(split_audio_paths)

		return audio_tags

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate captions for audio snippets.")
	parser.add_argument("config_file", help="Path to the file with configs for generation of captions")
	args = parser.parse_args()

	preprocessor = Preprocessor(args.config_file)
	print(preprocessor.preprocess_all())