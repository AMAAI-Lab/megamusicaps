import json
import argparse

# from essentia.standard import MonoLoader, TensorflowPredictMusiCNN, TensorflowPredictVGGish
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import numpy as np
# import matplotlib.pyplot as plt
# from glob import glob
# import multiprocessing
import os


parser = argparse.ArgumentParser(description="Extract tags from a dataset.")
parser.add_argument(
    "--index", type=int, default=0,
    help="Index of the split file to extract tags from."
)
args = parser.parse_args()

# audio_file = files[0]

def get_mtg_tags(embeddings,tag_model,tag_json,max_num_tags=5,tag_threshold=0.1):

    with open(tag_json, 'r') as json_file:
        metadata = json.load(json_file)
    model = TensorflowPredict2D(graphFilename=tag_model)
    predictions = model(embeddings)
    mean_act=np.mean(predictions,0)

    ind = np.argpartition(mean_act, -max_num_tags)[-max_num_tags:]

    tags=[]
    confidence_score=[]
    for i in ind:
        print(metadata['classes'][i] + str(mean_act[i]))
        if mean_act[i]>tag_threshold:
            tags.append(metadata['classes'][i])
            confidence_score.append(mean_act[i])

    ind=np.argsort(-np.array(confidence_score))
    tags = [tags[i] for i in ind]
    confidence_score=np.round((np.array(confidence_score)[ind]).tolist(),4).tolist()

    return tags, confidence_score

def get_voice_tag(embeddings,tag_model,tag_json):

    with open(tag_json, 'r') as json_file:
        metadata = json.load(json_file)

    model = TensorflowPredict2D(graphFilename=tag_model, output="model/Softmax")
    predictions = model(embeddings)
    mean_act=np.mean(predictions,0)

    ind = np.argmax(mean_act)
    tag=metadata['classes'][ind]

    return [tag], mean_act.tolist()

# files=glob('/666/TANGO/music-caps/data/*.wav')
# files=glob('/666/ds/fma_mini/*.mp3')
# ref_file='/666/TANGO/music-caps/eval_musiccaps.json'


emb_model="discogs-effnet-bs64-1.pb"

auto_model="mtt-discogs-effnet-1.pb"
auto_metadata="mtt-discogs-effnet-1.json"

mood_model="mtg_jamendo_moodtheme-discogs-effnet-1.pb"
mood_metadata="mtg_jamendo_moodtheme-discogs-effnet-1.json"

genre_model="mtg_jamendo_genre-discogs-effnet-1.pb"
genre_metadata="mtg_jamendo_genre-discogs-effnet-1.json"

inst_model="mtg_jamendo_instrument-discogs-effnet-1.pb"
inst_metadata="mtg_jamendo_instrument-discogs-effnet-1.json"

voice_model="voice_instrumental-discogs-effnet-1.pb"
voice_metadata="voice_instrumental-discogs-effnet-1.json"

gender_model="gender-discogs-effnet-1.pb"
gender_metadata="gender-discogs-effnet-1.json"

# source_folder='/666/ds/fma_large_cuts/'

# split_json_file='/666/ds/fma_mini/splitjson_{}.json'.format(args.index)
# output_json_file='/666/ds/fma_mini/fma_mini_tags_{}.json'.format(args.index)

split_json_file='/666/ds/fma_large_cuts/splitjson_{}.json'.format(args.index)
output_json_file='/666/ds/fma_large_cuts/split_tags_fma_large_{}.json'.format(args.index)

# path_to_data='/666/ds/fma_large_cuts/data'
# files=glob(source_folder+'*.mp3')
i=0
with open(split_json_file,'r') as split_json:
    with open(output_json_file,'w') as out_json:
        for row in split_json:
            a=json.loads(row)
            audio_file=a['location']
            # audio_file=os.path.join(path_to_data,os.path.basename(a['location']))
            #load audio
            #extract embs for all
            #get mood, genre, instru, voice -> if voice, get fem/m
            #if voice>0.8, both fem and male can? 0.55, if voice <0.8, fem/male unclear -> discard both
            #all into a list of lists
            audio = MonoLoader(filename=audio_file, sampleRate=16000, resampleQuality=4)()
            embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=emb_model, output="PartitionedCall:1")
            embeddings = embedding_model(audio)
            mood_tags, mood_cs = get_mtg_tags(embeddings,mood_model,mood_metadata,max_num_tags=5,tag_threshold=0.1)
            genre_tags, genre_cs = get_mtg_tags(embeddings,genre_model,genre_metadata,max_num_tags=4,tag_threshold=0.1)
            inst_tags, inst_cs = get_mtg_tags(embeddings,inst_model,inst_metadata,max_num_tags=7,tag_threshold=0.1)
            auto_tags, auto_cs = get_mtg_tags(embeddings,auto_model,auto_metadata,max_num_tags=8,tag_threshold=0.1)

            voice_tag, voice_prob = get_voice_tag(embeddings,voice_model,voice_metadata)

            if voice_prob[1]>0.5:
                gender_tag, gender_prob = get_voice_tag(embeddings,gender_model,gender_metadata)

                if np.max(np.array(gender_prob))<0.6:
                    if voice_prob[1]>0.8: # voice is definitely there, just not sure which?
                        gender_tag = ['female', 'male']
                    else: # voice might not be there?
                        gender_tag = []
                        gender_prob = []
            else:
                gender_tag=[]
                gender_prob=[]

            file_name=os.path.basename(audio_file).split('.')[0]
            new_row={}
            # new_row['name']=file_name
            new_row['location']=a['location']
            new_row['autotags']=[auto_tags, auto_cs]
            new_row['mood']=[mood_tags, mood_cs]
            new_row['genre']=[genre_tags, genre_cs]
            new_row['instrumentation']=[inst_tags, inst_cs]
            new_row['voice']=[voice_tag, voice_prob]
            new_row['gender']=[gender_tag, gender_prob]
            # new_row['old_captions']=a['old_captions']

        
            out_json.write(json.dumps(new_row) + '\n')
            print(i)
            i+=1