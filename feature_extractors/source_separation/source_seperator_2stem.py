from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import pandas as pd
import json

separator = Separator('spleeter:2stems')
audio_loader = AudioAdapter.default()

def separate_audio(audio_path, output_dir="/home/abhinaba_roy/mbench/data_seperated_train_only"):
    ##TODO
    waveform, _ = audio_loader.load(audio_path)
    separator.separate_to_file(audio_path, output_dir)

def process_files():
    with open('/home/abhinaba_roy/mbench/MusicBench_train_modified.json', 'r') as f:
        mbench_train_df = [json.loads(line) for line in f.readlines()]
    with open('/home/abhinaba_roy/mbench/MusicBench_test_A_modified.json', 'r') as f:
        mbench_test_A_df = [json.loads(line) for line in f.readlines()]
    for i, row in enumerate(mbench_train_df):
        print(f'working on {i}th row')
        eval_ = row['is_audioset_eval_mcaps']
        location = row['location']
        folder = location.split('/')[0]
        if eval_ is False:
            separate_audio(f'/home/abhinaba_roy/mbench/datashare/{location}',f'/home/abhinaba_roy/mbench/data_seperated_train_only/{folder}')
    for i, row in enumerate(mbench_test_A_df):
        print(f'working on {i}th row')
        eval_ = row['is_audioset_eval_mcaps']
        location = row['location']
        folder = location.split('/')[0]
        if eval_ is False:
            separate_audio(f'/home/abhinaba_roy/mbench/datashare/{location}',f'/home/abhinaba_roy/mbench/data_seperated_train_only/{folder}')
            
def main():
    process_files()

if __name__=='__main__':
    main()
    


    
    