import os
import json


num_splits=20
# split_json_path='/666/ds/music-caps/split_tags_ep'
split_json_path='/666/ds/fma_large_cuts/split_tags_fma_large'

output_json_path='/666/ds/fma_large_cuts/joint_tags_fma_large.json'

with open(output_json_path,'w') as out_json:
    for i in range(num_splits):
        with open(split_json_path+'_{}.json'.format(i),'r') as split_json:
            for row in split_json:
                a=json.loads(row)
                out_json.write(json.dumps(a) + '\n')

