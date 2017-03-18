from __future__ import print_function
import argparse
import json
import linecache
import nltk
import numpy as np
import os
import sys
from tqdm import tqdm
import random
import cPickle


from collections import Counter
from six.moves.urllib.request import urlretrieve

reload(sys)
sys.setdefaultencoding('utf8')
random.seed(42)
np.random.seed(42)

def data_from_json(filename):
    with open(filename) as data_file:
        data = []
        counter = 0
        for line in data_file:
            data.append(json.loads(line))
            counter += 1
    return data

def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    return map(lambda x:x.encode('utf8'), tokens)

def read_write_dataset(dataset, tier, prefix):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""
    count = 0
    skipped = 0

    print("came to read_write_dataset")

    with open(os.path.join(prefix, tier +'.passage.pkl'), 'wb') as passage_file,  \
         open(os.path.join(prefix, tier +'.question.pkl'), 'wb') as question_file,\
         open(os.path.join(prefix, tier +'.answer.pkl'), 'wb') as answer_file, \
         open(os.path.join(prefix, tier +'.type.pkl'), 'wb') as type_file:

        questions_global = []
        passages_global = []
        answers_global = []
        types_global = []

        for query_num in tqdm(range(len(dataset)), desc="Preprocessing {}".format(tier)):
            
            current_query = dataset[query_num]
            
            # Extract "query"
            question = current_query['query']
            question_tokens = tokenize(question)
            questions_global.append(question_tokens)

            # Extract "passages"
            passage_tuple_list = []
            for passage_num in range(len(current_query['passages'])):
                
                # Grab is_selected
                is_selected = current_query['passages'][passage_num]['is_selected']

                # Grab passage
                passage = current_query['passages'][passage_num]['passage_text']
                passage = passage.replace("''", '" ')
                passage = passage.replace("``", '" ')
                passage_tokens = tokenize(passage)

                # Create tuple
                passage_tuple = (is_selected, passage_tokens)
                passage_tuple_list.append(passage_tuple)
            passages_global.append(passage_tuple_list)

            # Extract "answers"
            answer_tokens_list = []
            for answer_num in range(len(current_query['answers'])):
                answer = current_query['answers'][answer_num]
                answer = answer.replace("''", '" ')
                answer = answer.replace("``", '" ')
                answer_tokens = tokenize(answer)
                answer_tokens_list.append(answer_tokens)
            answers_global.append(answer_tokens_list)

            # Extract "query_type"
            query_type = current_query['query_type']
            types_global.append(query_type)

            count += 1
            
        print (len(questions_global), " queries processed.")

        # Check if the number of entries are the same across all four categories of info extraction
        assert len(questions_global) == len(passages_global) == len(answers_global) == len(types_global)    

        # Checking code
        for i in range(100):
            print(i, " : ", passages_global[i])

        # Pickle
        try:
            # remove length restraint since we deal with it later
            print ("Pickling questions..")
            cPickle.dump(questions_global, question_file, -1)
            
            print ("Pickling passages..")
            cPickle.dump(passages_global, passage_file, -1)
            
            print ("Pickling answers..")  
            cPickle.dump(answers_global, answer_file, -1)
            
            print ("Pickling types..")
            cPickle.dump(types_global, type_file, -1)
        except Exception as e:
            skipped += 1

    print("Skipped {} question/answer pairs in {}".format(skipped, tier))
    return count, count


def save_files(prefix, tier, indices):
    
    with open(os.path.join(prefix, 'train.passage.pkl'), 'rb') as passage_file,  \
        open(os.path.join(prefix, 'train.question.pkl'), 'rb') as question_file,\
        open(os.path.join(prefix, 'train.answer.pkl'), 'rb') as answer_file, \
        open(os.path.join(prefix, 'train.type.pkl'), 'rb') as type_file:

        passages = cPickle.load(passage_file)
        questions = cPickle.load(question_file)
        answers = cPickle.load(answer_file)
        types = cPickle.load(type_file)

    new_passages = []
    new_questions = []
    new_answers = []
    new_types = []

    for i in indices:
        new_passages.append(passages[i])
        new_questions.append(questions[i])
        new_answers.append(answers[i])
        new_types.append(types[i])

    with open(os.path.join(prefix, tier + '.passage.pkl'), 'wb') as new_passage_file,  \
        open(os.path.join(prefix, tier + '.question.pkl'), 'wb') as new_question_file,\
        open(os.path.join(prefix, tier + '.answer.pkl'), 'wb') as new_answer_file, \
        open(os.path.join(prefix, tier + '.type.pkl'), 'wb') as new_type_file:

        cPickle.dump(new_passages, new_passage_file, -1)
        cPickle.dump(new_questions, new_question_file, -1)
        cPickle.dump(new_answers, new_answer_file, -1)
        cPickle.dump(new_types, new_type_file, -1)


def split_tier(prefix, train_percentage = 0.9, shuffle=False):
    # Get number of lines in file
    passage_filename = os.path.join(prefix, 'train' + '.passage.pkl')

    # Get the number of items
    with open(passage_filename, 'rb') as current_file:
        num_items = len(cPickle.load(current_file))
    
    # Get indices and split into two files
    indices_val = range(num_items)[int(num_items * train_percentage)::]
    if shuffle:
        np.random.shuffle(indices_val)
        print("Shuffling...")
    save_files(prefix, 'val', indices_val)
    
    indices_train = range(num_items)[:int(num_items * train_percentage)]
    if shuffle:
        np.random.shuffle(indices_train)
    save_files(prefix, 'train', indices_train)


if __name__ == '__main__':

    download_prefix = os.path.join("download", "marco")
    data_prefix = os.path.join("data", "marco")

    print("Downloading datasets into {}".format(download_prefix))
    print("Preprocessing datasets into {}".format(data_prefix))

    if not os.path.exists(download_prefix):
        os.makedirs(download_prefix)
    if not os.path.exists(data_prefix):
        os.makedirs(data_prefix)

    train_filename = "train_v1.1.json"
    dev_filename = "dev_v1.1.json"
    #test_filename = "test_public_v1.1.json"

    train_data = data_from_json(os.path.join(download_prefix, train_filename))
    train_num_questions, train_num_answers = read_write_dataset(train_data, 'train', data_prefix)
    split_tier(data_prefix, 0.90, shuffle=True)
    print("Processed {} questions and {} answers in train".format(train_num_questions, train_num_answers))

    dev_data = data_from_json(os.path.join(download_prefix, dev_filename))
    dev_num_questions, dev_num_answers = read_write_dataset(dev_data, 'dev', data_prefix)
    print("Processed {} questions and {} answers in dev".format(dev_num_questions, dev_num_answers))