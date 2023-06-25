import torch
from torch.utils.data import Dataset
import os
from Utils import utils
from Utils.utils import read_csv_file
import numpy as np
logger = utils.setup_logger(__name__, 'train.log')

section_delimiter = "========"


class Subject_Dataset(torch.utils.data.Dataset):

    def __init__(self, data_path, tokenizer):


        files = os.listdir(data_path)
        data_file_path = os.path.join(data_path,files[0])
        self.labels_class = utils.config['labels_class']
        #I want to read all the text files and creat a dataframe with a funtion here.
        #for the time being I am just reading the entire dataframe
        print("File path to read: ", data_file_path)
        self.df = read_csv_file(data_file_path)
        #print(self.df['category'])
        self.labels = [self.labels_class[label] for label in self.df['category']]
        self.texts = [tokenizer(str(text),
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in self.df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
