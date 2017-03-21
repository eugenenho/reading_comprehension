import numpy as np

from simple_configs import MAX_NUM_PASSAGES

# classifier results
def classifier_eval(preds, y, log_file = None):
	print 'lengths:', len(preds), len(y)
	assert len(preds) == len(y)

	num_correct = np.sum(np.equal(y, preds))
	accuracy = float(num_correct) / float(len(y))

	recalls = list()
	for i in range(MAX_NUM_PASSAGES):
		selected_pos_list = np.where(preds == i, 1, 0) # places where we chose this class
		actual_pos_list = np.where(y == i, 1, 0) #places where class i is true pos

		# this index never occurrs
		if np.sum(actual_pos_list) == 0: 
			recalls.append(0)
			continue

		true_positives_list = np.where(selected_pos_list + actual_pos_list == 2, 1, 0) # places where we were correct about this class
		num_true_pos = np.sum(true_positives_list)

		false_negatives_list = np.where( actual_pos_list - selected_pos_list == 1, 1, 0 )
		num_false_neg = np.sum(false_negatives_list)

		current_recall = float(num_true_pos) / float(num_true_pos + num_false_neg) # precision for this class

		recalls.append(current_recall)

	avg_recall = float(sum(recalls)) / float(len(recalls))

	precisions = list()
	for i in range(MAX_NUM_PASSAGES):
		selected_pos_list = np.where(preds == i, 1, 0)
		actual_pos_list = np.where(y == i, 1, 0)
		
		# this index never occurrs
		if np.sum(actual_pos_list) == 0:
			precisions.append(0)
			continue

		true_pos_list = np.where(actual_pos_list + selected_pos_list == 2, 1, 0)
		num_true_pos = np.sum(true_pos_list)

		false_pos_list = np.where( selected_pos_list - true_pos_list == 1, 1, 0)
		num_false_pos = np.sum(false_pos_list)

		current_precision = float(num_true_pos) / float(num_true_pos + num_false_pos) if num_true_pos + num_false_pos > 0 else float(num_true_pos)

		precisions.append(current_precision)

	avg_precision = float(sum(precisions)) / float(len(precisions))

	f1_score = 2.0 * avg_recall * avg_precision / (avg_recall + avg_precision) if avg_recall + avg_precision > 0 else 2.0 * avg_recall * avg_precision

	log_print(log_file, 'Evaluation Information for Classifier:')
	log_print(log_file, 'Accuracy: ' + str(accuracy))
	log_print(log_file, 'Precision: ' + str(avg_precision))
	log_print(log_file, 'Recall: ' + str(avg_recall))
	log_print(log_file, 'F1 Score: ' + str(f1_score))
	log_print(log_file, 'These are the results')

	return (accuracy, avg_recall, avg_precision, f1_score)

# helper method to either print to STDOUT or a log file
def log_print(log_file, message):
	if log_file is not None:
		log_file.write('\n' + str(message) )
	else:
		print message


if __name__ == "__main__":
	y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8])
	preds = np.array([9, 9, 9, 9, 9, 9, 9, 9, 9, 9])
	classifier_eval(preds, y)


















