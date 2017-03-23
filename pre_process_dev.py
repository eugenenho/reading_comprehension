from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import argparse
import cPickle

from six.moves import urllib

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin

_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_STR = b"<str>"
_END = b"<end>"
_START_VOCAB = [_PAD, _STR, _END, _SOS, _UNK]

PAD_ID = 0
STR_ID = 1
END_ID = 2
SOS_ID = 3
UNK_ID = 4

def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]

def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None):
    if not gfile.Exists(target_path):
        
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        
        with open(data_path, "rb") as data_file: 
            list_of_lists = cPickle.load(data_file)
        
        list_of_lists_ids = []
        counter = 0

        for single_list in list_of_lists:
            
            counter += 1
            if counter % 5000 == 0:
                print("tokenizing line %d" % counter)
            
            token_id_list =[]

            for item in single_list: # Each item representing one passage (i.e. list of tokens)
                
                # This applies when dealing with passags.pkl or answers.pkl (some queries has >1 answers)
                # When dealing with passages.pkl: [ [ [token, .., token],.., [token, .., token]  ]]
                #                                      ^ Each passage 
                #                                   ^ Each query has ~10 passages
                #                                 ^ list of all queries' passages
                if(isinstance(item, list)): # Each item representing one answer
                    intermediate_step = [vocab.get(w, UNK_ID) for w in item] # each w representing one token
                    token_id_list.append(intermediate_step)
                elif(isinstance(item, tuple)): # Each item representing one passage tuple 
                    is_selected, passage_tokens = item
                    intermediate_step = [vocab.get(w, UNK_ID) for w in passage_tokens] # each passage_tuple[1] representing one token (passage_tuple[0] is is_selected)
                    token_id_list.append(intermediate_step)
                else:       
                    token_id_list.append(vocab.get(item, UNK_ID))
            list_of_lists_ids.append(token_id_list)
        
        print("processing: ", data_path)
        # print("list_of_lists_ids", list_of_lists_ids)

        with open(target_path, "wb") as target_file:
            cPickle.dump(list_of_lists_ids, target_file, -1)


if __name__ == "__main__":
    data_to_token_ids("./data/marco/dev.passage.pkl", "./data/marco/dev.ids.passage.pkl", './data/marco/vocab.dat')
    data_to_token_ids("./data/marco/dev.question.pkl", "./data/marco/dev.ids.question.pkl", './data/marco/vocab.dat')
    data_to_token_ids("./data/marco/dev.answer.pkl", "./data/marco/dev.ids.answer.pkl", './data/marco/vocab.dat')



























