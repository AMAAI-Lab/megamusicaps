from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import os
import shutil
import time
from pydub import AudioSegment

separator = Separator('spleeter:4stems')

audio_loader = AudioAdapter.default()

def separate_audio(audio_path, output_dir="/proj/megamusicaps/files/temp/"):

    waveform, _ = audio_loader.load(audio_path)

    # Perform the separation :
    separator.separate_to_file(audio_path, output_dir)

    # Move files from the subdirectory to the specified output_dir
    subdirectory = os.path.join(output_dir, os.path.basename(audio_path)[:-4])  # Extract file name without extension
    for stem in ["vocals", "drums", "bass", "other"]:
        stem_file_wav = os.path.join(subdirectory, f"{stem}.wav")
        stem_file_mp3 = os.path.join(output_dir, f"{stem}.mp3")
        
        # Convert WAV to MP3
        audio_segment = AudioSegment.from_wav(stem_file_wav)
        print("Converting ", stem_file_wav, "to", stem_file_mp3)
        audio_segment.export(stem_file_mp3, format="mp3")
        
        # Remove the WAV file
        os.remove(stem_file_wav)

    # Remove the empty subdirectory
    os.rmdir(subdirectory)

    prediction = {
        "raw": audio_path,
        "vocals": os.path.join(output_dir, "vocals.mp3"),
        "drums": os.path.join(output_dir, "drums.mp3"),
        "bass": os.path.join(output_dir, "bass.mp3"),
        "other": os.path.join(output_dir, "other.mp3")
    }

    return prediction
