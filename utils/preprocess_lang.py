import pickle
import os
import json
from tqdm import tqdm
import nltk

def gen_word2id(lang_train_file, lang_test_file):
    lang_train_data = list(open(lang_train_file))
    lang_test_data = list(open(lang_test_file))
    lang_data = lang_train_data + lang_test_data
    start_index = 1
    word2id = {}
    for item in tqdm(lang_data):
        query = item.strip().split("##")[-1].replace('.', '')
        query_words = nltk.word_tokenize(query)
        for word in query_words:
            if word not in word2id:
                word2id[word] = start_index
                start_index += 1
    json.dump(word2id, open('../data/dataset/Charades/Charades_word2id.json', 'w'), indent=2)


if __name__ == '__main__':
    lang_train_file = '../data/charades_sta_train.txt'
    lang_test_file = '../data/charades_sta_test.txt'
    gen_word2id(lang_train_file, lang_test_file)





