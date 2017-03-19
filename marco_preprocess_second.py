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

def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("data", "marco")
    glove_dir = os.path.join("download", "dwr")
    source_dir = os.path.join("data", "marco")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=50, type=int)
    return parser.parse_args()


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


def process_glove(args, vocab_list, save_path, size=4e5):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if not gfile.Exists(save_path + ".npz"):
        glove_path = os.path.join(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        glove = np.zeros((len(vocab_list), args.glove_dim))
        not_found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                elif word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                elif word.lower() in vocab_list:
                    idx = vocab_list.index(word.lower())
                    glove[idx, :] = vector
                elif word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                else:
                    not_found += 1
        found = size - not_found
        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:

            with open(path, mode="rb") as f:
                list_of_lists = cPickle.load(f)
                counter = 0

                for single_list in list_of_lists:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    
                    for item in single_list: 
                        # This applies when dealing with answers.pkl (some queries has >1 answers)
                        # When dealing with answers.pkl: [ [ [token, .., token],.., [token, .., token]  ]]
                        #                                      ^ Each answer 
                        #                                   ^ Each query has >=1 answers
                        #                                 ^ list of all queries' answers
                        if(isinstance(item, list)): # Each item representing one answer
                        
                            for w in item: # Each w representing one token in answer
                                if w in vocab:
                                    vocab[w] += 1
                                else:
                                    vocab[w] = 1
                        
                        # This applies when dealing with passages.pkl (some queries has >1 answers)
                        # When dealing with passages.pkl: [ [ (0, [token, .., token]),.., (1, [token, .., token])  ]]
                        #                                      ^ Each passage 
                        #                                   ^ Each query has >=1 passages
                        #                                 ^ list of all queries' passages
                        elif(isinstance(item, tuple)): # Each item representing one passage tuple 
                            is_selected, passage_tokens = item
                            for w in passage_tokens: # Each w representing one token in a passage 
                                if w in vocab:
                                    vocab[w] += 1
                                else:
                                    vocab[w] = 1

                        # If item is word:
                        else:
                            if item in vocab:
                                vocab[item] += 1
                            else:
                                vocab[item] = 1    

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        # print(vocab_list)

        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")

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
            

if __name__ == '__main__':
    args = setup_args()
    vocab_path = pjoin(args.vocab_dir, "vocab.dat")
    train_path = pjoin(args.source_dir, "train")
    valid_path = pjoin(args.source_dir, "val")

    create_vocabulary(vocab_path,
                      [pjoin(args.source_dir, "train.passage.pkl"),
                       pjoin(args.source_dir, "train.question.pkl"),
                       pjoin(args.source_dir, "train.answer.pkl"),
                       pjoin(args.source_dir, "val.passage.pkl"),
                       pjoin(args.source_dir, "val.question.pkl"),
                       pjoin(args.source_dir, "val.answer.pkl")])
    
    print ("create_vocabulary done!!")
    vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.dat"))

    # ======== Trim Distributed Word Representation =======
    # If you use other word representations, you should change the code below

    # process_glove(args, rev_vocab, args.source_dir + "/glove.trimmed.{}".format(args.glove_dim))

    # ======== Creating Dataset =========
    # We created our data files seperately
    # If your model loads data differently (like in bulk)
    # You should change the below code

    p_train_ids_path = train_path + ".ids.passage.pkl"
    q_train_ids_path = train_path + ".ids.question.pkl"
    a_train_ids_path = train_path + ".ids.answer.pkl"
    data_to_token_ids(train_path + ".passage.pkl", p_train_ids_path, vocab_path)
    data_to_token_ids(train_path + ".question.pkl", q_train_ids_path, vocab_path)
    data_to_token_ids(train_path + ".answer.pkl", a_train_ids_path, vocab_path)

    p_valid_ids_path = valid_path + ".ids.context.pkl"
    q_valid_ids_path = valid_path + ".ids.question.pkl"
    a_valid_ids_path = valid_path + ".ids.answer.pkl"
    data_to_token_ids(valid_path + ".passage.pkl", p_valid_ids_path, vocab_path)
    data_to_token_ids(valid_path + ".question.pkl", q_valid_ids_path, vocab_path)
    data_to_token_ids(valid_path + ".answer.pkl", a_valid_ids_path, vocab_path)


    # ======== Testing code ==========
    
    print ("Testing the output pkl : train\n")

    with open(p_train_ids_path, 'rb') as passage_ids_file, \
        open(q_train_ids_path, 'rb') as question_ids_file, \
        open(a_train_ids_path, 'rb') as answer_ids_file, \
        open(train_path + ".passage.pkl", 'rb') as passage_file, \
        open(train_path + ".question.pkl", 'rb') as question_file, \
        open(train_path + ".answer.pkl", 'rb') as answer_file:
        questions_ids = cPickle.load(question_ids_file)
        answers_ids = cPickle.load(answer_ids_file)
        passages_ids = cPickle.load(passage_ids_file)
        questions = cPickle.load(question_file)
        answers = cPickle.load(answer_file)
        passages = cPickle.load(passage_file)
        
    test_limit = 10
    
    for i in range(len(answers_ids)):
        if i >= test_limit: break
        print(i, "\t", questions[i])
        print(i, "\t", questions_ids[i])

    for i in range(len(answers_ids)):
        if i >= test_limit: break
        print(i, "\t", answers[i])
        print(i, "\t", answers_ids[i])

    for i in range(len(passages_ids)):
        if i >= test_limit: break
        for j in range(len(passages_ids[i])):
            print(i, "-", j, "\t", passages[i][j])
            print(i, "-", j, "\t", passages_ids[i][j])