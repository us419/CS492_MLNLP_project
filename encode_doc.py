import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from src.utils import get_max_lengths, get_evaluation
import pickle
import os
from tqdm import tqdm

if os.path.isfile("Yelp_texts.txt"):
    with open("Yelp_texts.txt", "rb") as fp:   # Unpickling
        temp = pickle.load(fp)
        fp.close()

else:
    train_set = "data/yahoo_answers_csv/train.csv"
    test_set = "data/yahoo_answers_csv/test.csv"
    max_word_length, max_sent_length = get_max_lengths(train_set)
    dict_path = "data/glove.6B.200d.txt"
    glove_dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                            usecols=[0]).values
    glove_dict = [word[0] for word in glove_dict]

    texts, labels = [], []
    with open(test_set, encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(tqdm(reader)):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
                document_encode = [
                    [glove_dict.index(word) if word in glove_dict else -1 for word in word_tokenize(text=sentences)] for sentences
                    in
                    sent_tokenize(text=text)]

                for sentences in document_encode:
                    if len(sentences) < max_word_length:
                        extended_words = [-1 for _ in range(max_word_length - len(sentences))]
                        sentences.extend(extended_words)

                if len(document_encode) < max_sent_length:
                    extended_sentences = [[-1 for _ in range(max_word_length)] for _ in
                                        range(max_sent_length - len(document_encode))]
                    document_encode.extend(extended_sentences)

                document_encode = [sentences[:max_word_length] for sentences in document_encode][
                                :max_sent_length]

                document_encode = np.stack(arrays=document_encode, axis=0)
                document_encode += 1
                
            label = int(line[0]) - 1
            texts.append(document_encode.astype(np.int64))
            labels.append(label)

    with open("Test_yahoo_texts.txt", "wb") as fp:   #Pickling
        pickle.dump(texts, fp)
        fp.close()

    with open("Test_yahoo_labels.txt", "wb") as fp:   #Pickling
        pickle.dump(labels, fp)
        fp.close()
