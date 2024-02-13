import json
import pandas as pd
from torch.utils.data import Dataset


class InstructDataset(Dataset):
    def __init__(self, filename="", num_examples=-1, data=None):
        
        if data == None:
            data = [json.loads(line) for line in open(filename).readlines()]
        
        if num_examples != -1:
            data = data[:num_examples]
            
        self.paths = [instance["audio"] for instance in data]
        self.questions = [instance["question"] for instance in data]
        self.answers = [instance["answer"] for instance in data]

    def __len__(self):
        return len(self.paths)

    def get_num_instances(self):
        return len(self.paths)

    def __getitem__(self, index):
        s1, s2, s3 = self.paths[index], self.questions[index], self.answers[index]
        return s1, s2, s3

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]