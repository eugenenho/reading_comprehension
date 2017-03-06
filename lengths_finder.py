import cPickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

train_questions = cPickle.load( open("./data/marco/train.ids.question.pkl", "rb" ) )

question_length = [len(train_questions[i]) for i, _ in enumerate(train_questions)]
question_length = np.asarray(question_length)

plt.hist(question_length)
plt.title("Histogram of question lengths")
plt.show()

train_passages = cPickle.load( open("./data/marco/train.ids.passage.pkl", "rb" ) )

passage_set_length = list()
i = 0
for row in tqdm(train_passages):
	tot_len = 0
	for p in row:
		tot_len += len(p)
	passage_set_length.append(tot_len)

passage_length = np.asarray(passage_set_length)

plt.hist(passage_length)
plt.title("Histogram of all passage sum lengths")
plt.show()

