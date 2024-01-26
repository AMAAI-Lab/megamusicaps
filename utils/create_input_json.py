import os
import json

def get_mp3_files(directory):
    mp3_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".mp3"):
                mp3_files.append(os.path.join(root, file))
    return mp3_files

def create_json_file(directory_paths, output_json_file):
    data = []
    for directory_path in directory_paths:
        mp3_files = get_mp3_files(directory_path)
        data.extend([{"location": file} for file in mp3_files])

    with open(output_json_file, 'w') as json_file:
        for entry in data:
            json.dump(entry, json_file)
            json_file.write('\n')

if __name__ == "__main__":
    # Example usage:
    input_directories = ["/proj/megamusicaps/samples/fma_top_downloads/", "/proj/megamusicaps/samples/generic_pop_audio/", "/proj/megamusicaps/samples/mtg_jamendo/"]
    output_json_file = "/proj/megamusicaps/files/audio_files.json"

    create_json_file(input_directories, output_json_file)
