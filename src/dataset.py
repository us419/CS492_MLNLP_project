"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pickle


class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()
        texts, labels = [], []
        if (data_path):
            with open("Train_yahoo_texts.txt", "rb") as fp:
                texts = pickle.load(fp)
                fp.close()
            
            with open("Train_yahoo_labels.txt", "rb") as fp:
                labels = pickle.load(fp)
                fp.close()

        else:
            with open("Test_yahoo_texts.txt", "rb") as fp:
                texts = pickle.load(fp)
                fp.close()
            
            with open("Test_yahoo_labels.txt", "rb") as fp:
                labels = pickle.load(fp)
                fp.close()
        self.texts = texts
        self.labels = labels
        # self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
        #                         usecols=[0]).values
        # self.dict = [word[0] for word in self.dict]
        # self.max_length_sentences = max_length_sentences
        # self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))
        print("num classes : ", self.num_classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]

        return text, label


if __name__ == '__main__':
    test = MyDataset(data_path="../data/test.csv", dict_path="../data/glove.6B.50d.txt")
    print (test.__getitem__(index=1)[0].shape)
