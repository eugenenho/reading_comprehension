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


if __name__ == '__main__':

	data_prefix = os.path.join("data", "marco")
	with open(os.path.join(data_prefix, 'train.passage.pkl'), 'rb') as passage_file:
    		p = cPickle.load(passage_file)

	    
	for i in range(100):
		print(i, " : ", p[i])
    
