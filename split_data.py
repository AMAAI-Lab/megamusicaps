import glob
import os
import shutil
import json
import numpy as np
#lets try json first only


#from glob to jsons
# filespath='/666/ds/fma_mini'
filespath='/666/ds/fma_large_cuts'

# files=glob.glob(filespath+'/*.mp3')
files=glob.glob(filespath+'/*/*.wav')

out_json_path='/666/ds/fma_large_cuts/splitjson'
num_splits=20

max_index=np.ceil(len(files)/num_splits)
end_index=len(files)
cur_index=0
for i in range(num_splits):

    with open(out_json_path+'_{}.json'.format(i),'w') as out_json:
        for j in range(int(max_index)):
            new_row={}
            if cur_index==end_index:
                break
            new_row['location']=files[cur_index]
            out_json.write(json.dumps(new_row) + '\n')


            cur_index+=1


#from json to jsons
# big_json='/666/ds/music-caps/musiccaps_ep.json'
# split_json_path='/666/ds/music-caps/split_musiccaps_ep'

# num_splits=20
# end_index=5479
# max_index=np.ceil(end_index/num_splits)
# cur_index=0
# cur_file=-1
# big_write_list=[]
# write_list=[]
# with open(big_json,'r') as big:
#     for row in big:
#         if np.mod(cur_index,max_index)==0:
#             cur_file+=1
#             if cur_file>0:
#                 big_write_list.append(write_list)
#             write_list=[]
#         a=json.loads(row)
#         write_list.append(a)
#         cur_index+=1
#         if cur_index==end_index:
#                 big_write_list.append(write_list)



# print(write_list)
# cur_index=0
# for i in range(num_splits):
#     with open(split_json_path+'_{}.json'.format(cur_index),'w') as split_json:
#         for row in big_write_list[i]:

#             split_json.write(json.dumps(row) + '\n')
#     cur_index+=1


# with open()
