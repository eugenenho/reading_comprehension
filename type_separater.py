import numpy as np 
import cPickle
from collections import defaultdict


def group_types():
	types_list = cPickle.load(open('./data/marco/dev.type.pkl',"rb"))
	preds_file = open('./data/dev_preds.json', 'r')
	ground_truth_file = open('./data/dev_ground_truth.json', 'r')

	separated_preds = defaultdict(lambda: [])
	separated_truths = defaultdict(lambda: [])

	for curr_type, pred, ground_truth in zip(types_list, preds_file, ground_truth_file):
		separated_preds[curr_type].append(pred)
		separated_truths[curr_type].append(ground_truth)
	preds_file.close()
	ground_truth_file.close()

	types_set = set(types_list)
	for t in types_set:
		f1 = open('./data/sep_preds/'+t+'_pred.json', 'w')
		f2 = open('./data/sep_preds/'+t+'_truth.json', 'w')

		for p in separated_preds[t]:
			f1.write(p)
		f1.close()

		for gt in separated_truths[t]:
			f2.write(gt)
		f2.close()


if __name__ == "__main__":
	group_types()