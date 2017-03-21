# reading_comprehension
CS 224n project on Reading Comprehension / Question-Answering using Microsoft MARCO dataset

# Required folder structure
Let's call the main project folder "rc_project" (where the files in this repo goes to)
1. Download Glove "http://nlp.stanford.edu/data/glove.6B.zip" and put it in "rc_project/download/dwr" (create this folder)
2. Download training and dev datasets for Marco and put them in "rc_project/download/marco" (create this folder)

#Setup
pip install tqdm
Go to Keras and intall it separately
Go to Recurrentshop and install it separately
Go to Seq2seq and install it
pip install h5py (for save model)
(make sure file structure includes data and download as described in github)
For marco script: sudo python -m spacy.en.download --force all

# Preprocesing steps:
Run marco_preprocessing.py first, then marco_preprocessing_second.py

# Simple Model:
This simple model is the baseline module for the project.  It is a basic seq2seq model where the question and passages are all combined with simple concatination.  Each row is then padded to a length of 1000 word embeddings, and then trained.

# Data And Models:
Data and Models are too large to save on github.  They can be found on the dropbox we have set up:
	https://www.dropbox.com/sh/g9bb5ralrmlj0lz/AACdz6TGCxqW_4acMD1cnmAza?dl=0