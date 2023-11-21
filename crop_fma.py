import os
import json
import glob
import soundfile as sf
import librosa

data_folder='/666/ds/fma_large'
out_folder='/666/ds/fma_large_cuts'
audio_files=glob.glob(data_folder+'/*/*.mp3')

sr=16000
crop_len=10
i=0
j=0
for aufil in audio_files:
    try:
        y, srrr = librosa.load(aufil,sr=sr)
    except:
        j+=1
        continue

    if len(y)<sr*crop_len:
        continue
    # data = librosa.resample(y, srrr, sr)
    data=y
    data1 = data[0:sr*crop_len]
    data2 = data[sr*crop_len:2*sr*crop_len]
    data3 = data[2*sr*crop_len:]





    if len(data3)>sr*crop_len:
        data3 = data3[0:sr*crop_len]


    output_path1=os.path.join(out_folder,os.path.dirname(aufil).split('/')[-1],os.path.basename(aufil).split('.')[0]+'_1.wav')#or basename
    os.makedirs(os.path.dirname(output_path1),exist_ok=True)
    output_path2=os.path.join(out_folder,os.path.dirname(aufil).split('/')[-1],os.path.basename(aufil).split('.')[0]+'_2.wav')#or basename
    output_path3=os.path.join(out_folder,os.path.dirname(aufil).split('/')[-1],os.path.basename(aufil).split('.')[0]+'_3.wav')#or basename

    sf.write(output_path1,data1,sr)

    if len(data2)<sr*crop_len/2:
        continue
    else:
        sf.write(output_path2,data2,sr)

    if len(data3)<sr*crop_len/2:
        continue
    else:
        sf.write(output_path3,data3,sr)


    print(i)
    i+=1

print('failed files: '+str(j))