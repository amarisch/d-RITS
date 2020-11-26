# Computes the Classificatiton performance of baseline models:
#	Random Classifier
#	Single Layer Perceptron
#	Random Forest Classifier

# python3 baseline.py --method rf --data testinference3_drop3 --savepath mybaseline
# python3 baseline.py --method svm --data testinference3_drop3 --savepath mybaseline
# python3 baseline.py --data 6_10/test_xcov1 --savepath 6_10/test_xcov1
# python3 baseline.py --data patdata24_all5_seq20_feat65 --savepath test
# python3 baseline.py --data patdata24_new2 --savepath 7_20/baseline --valpath 7_20 --cv 3 --useval
# python3 baseline.py --data patdata24 --savepath 7_24/baseline --valpath 7_24 --cv 3 --useval
# python3 baseline.py --data patdata48 --savepath 7_24/v48/baseline --valpath 7_24/v48 --cv 3 --useval


import numpy as np
import pandas as pd
import argparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Perceptron

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import json
import shap

import pickle

import warnings

parser= argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--savepath', type=str)
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--useval', action='store_true')
parser.add_argument('--valpath', type=str)
args = parser.parse_args()
if args.valpath is None:
	args.valpath = args.savepath

if not os.path.exists(args.savepath):
    try:
        os.mkdir(args.savepath)
    except OSError:
        print('Savepath folder creation error')
        exit()

for i in range(1,4):
	if not os.path.exists("{}/run{}".format(args.savepath, i)):
	    try:
	        os.mkdir("{}/run{}".format(args.savepath, i))
	    except OSError:
	        print('Savepath folder creation error')
	        exit()


f = open(os.path.join(args.savepath, 'baseline_out'), 'w')

def plot(recall, precision, auprc, method, n_classes=5):
    # setup plot details
    # colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    colors = cycle(['red', 'orange', 'yellow', 'green', 'blue'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='grey',linestyle='--', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(auprc["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color,marker='.', lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, auprc[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.00])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall of {}'.format(method))
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig("{}.png".format(method))
    plt.show()

topfeats = ['vm1', 'vm2', 'vm3', 'vm4', 'vm5', 'vm6', 'vm7', 'vm8', 'vm9', 'vm10', 'vm11', 'vm13', 'vm14', 'vm15', 'vm20', 'vm22', 'vm29', 'vm32', 'pm41', 'pm42', 'vm58', 'vm62', 'vm63', 'vm64', 'vm131', 'vm132', 'vm133', 'vm134', 'vm135', 'vm136', 'vm137', 'vm138', 'vm139', 'vm140', 'vm141', 'vm142', 'vm143', 'vm144', 'vm145', 'vm148', 'vm149', 'vm150', 'vm151', 'vm153', 'vm154', 'vm155', 'vm156', 'vm160', 'vm161', 'vm162', 'vm163', 'vm164', 'vm165', 'vm166', 'vm172', 'vm173', 'vm174', 'vm175', 'vm176', 'vm178', 'vm180', 'vm183', 'vm185', 'vm188', 'vm194']
featdata = pd.read_csv("mimic-varref.csv")

def get_feature(num):
    return featdata.loc[featdata['HIRID_METAID']== topfeats[num],'HIRID_LABEL'].iloc[0]

def get_label_weight_ratio(labels):
	bettercounts = np.sum(labels, axis=0)
	return np.round(max(bettercounts)/bettercounts).tolist()

def get_label_weight_ratio_ova(labels):
	bettercounts = np.sum(labels, axis=0)
	return np.round(sum(bettercounts)/bettercounts).tolist()

def train_val_split(X, Y, train_ratio):
    if train_ratio > 1 or train_ratio <= 0:
        print('Training set ratio has to be smaller than 1, readjusted to 0.8.')
        train_ratio = 0.8
    num_samples = len(Y)
    indices = np.arange(num_samples)
    val_idx = np.random.choice(indices, int(num_samples*(1-train_ratio)))
    train_idx = np.setdiff1d(indices, val_idx)
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]

# “stratified”: generates predictions by respecting the training set’s class distribution.
# “most_frequent”: always predicts the most frequent label in the training set.
# “uniform”: generates predictions uniformly at random.


def evaluation(model, x_val, y_val):
	preds = model.predict_proba(x_val)
	preds = np.array(preds)

	# https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
	n_classes = y_val.shape[1]
	precision = dict()
	recall = dict()
	average_precision = dict()
	auprc = dict()
	for i in range(n_classes):
		predicted = preds[:, i]
		actual = y_val[:, i]
		precision[i], recall[i], _ = precision_recall_curve(actual, predicted)
		average_precision[i] = average_precision_score(actual, predicted)
		#         f1 = f1_score(actual, predicted)
		auprc[i] = auc(recall[i], precision[i])
		aps = average_precision_score(actual, predicted)
		print('Class %d ROC PRC= %.3f' % (i, auprc[i]))
		#     print('\tF1= %.3f' % (f1))
		print('\tAP Score= %.3f' % (aps))

	# plot curve https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
	auroc = roc_auc_score(y_val, preds)
	# A "micro-average": quantifying score on all classes jointly
	precision["micro"], recall["micro"], _ = precision_recall_curve(y_val.ravel(), preds.ravel())
	auprc["micro"] = auc(recall["micro"], precision["micro"])
	average_precision["micro"] = average_precision_score(y_val, preds, average="micro")
	print('AUPRC over all classes: {0:0.2f}'.format(auprc["micro"]))
	#     print('AP score over all classes: {0:0.2f}'.format(average_precision["micro"]))
	return recall, precision, auprc

types = ['RandomForest', 'RandomClassifier', 'SingleLayerPerceptron']
color = {'RandomForest':'red', 'RandomClassifier':'orange', 'SingleLayerPerceptron':'green', 'RNN':'blue', 'VRAE':'purple'}

def cross_validate(clf, X, y, cv, clfname='classifier'):
	n_classes = 5
	arr = np.arange(y.shape[0])
	cv_val_idx = []
	cv_train_idx = []
	scores = {}
	prc = {}
	scores['precision'] = {}
	scores['recall'] = {}
	for i in range(cv):
		scores['precision'][i] = {}
		scores['recall'][i] = {}
	# scores['precision_std'] = {}
	# scores['recall80'] = np.zeros(5)
	# scores['recall85'] = np.zeros(5)
	# scores['recall90'] = np.zeros(5)
	if args.useval:
		val_idx = np.load('{}/val1/run_val_idx.npy'.format(args.valpath))
		cv_val_idx.append(val_idx)
		cv_train_idx.append(np.setdiff1d(arr, val_idx))
		val_idx = np.load('{}/val2/run_val_idx.npy'.format(args.valpath))
		cv_val_idx.append(val_idx)
		cv_train_idx.append(np.setdiff1d(arr, val_idx))
		val_idx = np.load('{}/val3/run_val_idx.npy'.format(args.valpath))
		cv_val_idx.append(val_idx)
		cv_train_idx.append(np.setdiff1d(arr, val_idx))
	else:
		np.random.shuffle(arr)
		val_size = int(y.shape[0] * 0.2)
		for i in range(0,cv):
			val_idx = arr[val_size*i:val_size*(i+1)]
			cv_val_idx.append(val_idx)
			cv_train_idx.append(np.setdiff1d(arr, val_idx))

	weight = get_label_weight_ratio_ova(y) # 1x5 vector
	accuracy = np.zeros((cv, n_classes))
	balanced_accuracy = np.zeros((cv, n_classes))
	f1score = np.zeros((cv, n_classes))
	average_precision = np.zeros((cv, n_classes))
	t_accuracy = np.zeros((cv, n_classes))
	t_balanced_accuracy = np.zeros((cv, n_classes))
	t_f1score = np.zeros((cv, n_classes))
	t_average_precision = np.zeros((cv, n_classes))
	for i in range(0, cv):

		clf.fit(X[cv_train_idx[i]], y[cv_train_idx[i]])
		y_true = y[cv_val_idx[i]]
		y_pred = clf.predict(X[cv_val_idx[i]])
		y_prob = np.array(clf.predict_proba(X[cv_val_idx[i]]))
		if (y_prob.shape[0] == n_classes):
			y_prob = y_prob[:,:,1]
			y_prob = np.transpose(y_prob)

		t_y_true = y[cv_train_idx[i]]
		t_y_pred = clf.predict(X[cv_train_idx[i]])
		t_y_prob = np.array(clf.predict_proba(X[cv_train_idx[i]]))
		if (t_y_prob.shape[0] == n_classes):
			t_y_prob = t_y_prob[:,:,1]
			t_y_prob = np.transpose(t_y_prob)

		for cat in range(0, n_classes):
			val_weight = [weight[cat] if i==1 else 1 for i in y_true[:,cat]]
			accuracy[i][cat] = accuracy_score(y_true[:,cat], y_pred[:,cat])
			balanced_accuracy[i][cat] = accuracy_score(y_true[:,cat], y_pred[:,cat], sample_weight=val_weight)
			f1score[i][cat] = f1_score(y_true[:,cat], y_pred[:,cat])
			average_precision[i][cat] = average_precision_score(y_true[:,cat], y_prob[:,cat])
			pre, recall, _ = precision_recall_curve(y_true[:,cat], y_prob[:,cat])

			scores['precision'][i][cat] = pre.tolist()
			scores['recall'][i][cat] = recall.tolist()

			t_weight = [weight[cat] if i==1 else 1 for i in t_y_true[:,cat]]
			t_accuracy[i][cat] = accuracy_score(t_y_true[:,cat], t_y_pred[:,cat])
			t_balanced_accuracy[i][cat] = accuracy_score(t_y_true[:,cat], t_y_pred[:,cat], sample_weight=t_weight)
			t_f1score[i][cat] = f1_score(t_y_true[:,cat], t_y_pred[:,cat])
			t_average_precision[i][cat] = average_precision_score(t_y_true[:,cat], t_y_prob[:,cat])

		if (clfname=='RandomForest'):
			np.save("{}/run{}/rf_pred".format(args.savepath, i+1), clf.predict(X))
			filename = "{}/run{}/rfmodel.sav".format(args.savepath, i+1)
			pickle.dump(clf, open(filename, 'wb'))
			explainer = shap.TreeExplainer(clf)
			expected_value = explainer.expected_value
			col_name=[]
			for z in range(X.shape[1]):
				time = int(z/65) + 1
				feat = get_feature(z%65)
				col_name.append('{}_{}'.format(feat, time))
			X_display = pd.DataFrame(X, columns=col_name)
			# print(X_display.shape)
			Xtest_display = X_display.iloc[cv_val_idx[i]]
			shap_values = explainer.shap_values(Xtest_display[:256],check_additivity=False)[1::2]
			rf_svals = []
			for cat in range(5):
				rf_svals.append(np.sum(abs(shap_values[cat]),axis=0).argsort()[::-1])
			np.save("{}/run{}/rf_idx".format(args.savepath, i+1), np.array(rf_svals))

	scores['test_accuracy'] = np.mean(np.array(accuracy), axis=0).tolist()
	scores['test_balanced_accuracy'] = np.mean(np.array(balanced_accuracy), axis=0).tolist()
	scores['test_f1'] = np.mean(np.array(f1score), axis=0).tolist()
	scores['test_average_precision'] = np.mean(np.array(average_precision), axis=0).tolist()

	# scores['train_accuracy'] = np.mean(np.array(t_accuracy), axis=0).tolist()
	# scores['train_balanced_accuracy'] = np.mean(np.array(t_balanced_accuracy), axis=0).tolist()
	# scores['train_f1'] = np.mean(np.array(t_f1score), axis=0).tolist()
	# scores['train_average_precision'] = np.mean(np.array(t_average_precision), axis=0).tolist()

	scores['test_accuracy_std'] = np.std(np.array(accuracy), axis=0).tolist()
	scores['test_balanced_accuracy_std'] = np.std(np.array(balanced_accuracy), axis=0).tolist()
	scores['test_f1_std'] = np.std(np.array(f1score), axis=0).tolist()
	scores['test_average_precision_std'] = np.std(np.array(average_precision), axis=0).tolist()

	# scores['train_accuracy_std'] = np.std(np.array(t_accuracy), axis=0).tolist()
	# scores['train_balanced_accuracy_std'] = np.std(np.array(t_balanced_accuracy), axis=0).tolist()
	# scores['train_f1_std'] = np.std(np.array(t_f1score), axis=0).tolist()
	# scores['train_average_precision_std'] = np.std(np.array(t_average_precision), axis=0).tolist()

	scores['name'] = clfname


	if clfname=="RandomClassifier":
		for i in range(5):
			for j in range(cv):
				scores['precision'][j][i] = [scores['test_average_precision'][i], scores['test_average_precision'][i]]
				scores['recall'][j][i] = [0,1]

	recall = {}
	precision = {}
	precision_std = {}
	for cat in range(5):
		df_ls = []
		for i in range(3):
			df = pd.DataFrame(columns=np.round(scores['recall'][i][cat], 6))
			df.loc[len(df)] = scores['precision'][i][cat]
			df = df.loc[:,~df.columns.duplicated()]
			df_ls.append(df)
		res = pd.merge(df_ls[0],df_ls[1], how="outer")
		res = pd.merge(res,df_ls[2], how="outer")
		if not 0.80 in res.columns: res.insert(len(res.columns), 0.80, np.NaN)
		if not 0.85 in res.columns: res.insert(len(res.columns), 0.85, np.NaN)
		if not 0.90 in res.columns: res.insert(len(res.columns), 0.90, np.NaN)
		res.sort_index(axis=1, inplace=True)
		res = res.interpolate(axis=1)
		mean = res.mean()
		std = res.std()
		res.loc[len(res)]= mean
		res.loc[len(res)]= std

		if clfname == "RandomClassifier":
			precision[cat] = res.loc[0].tolist()
			precision_std[cat] = np.zeros(2).tolist()
			recall[cat] = np.array(res.columns).tolist()
		else:
			precision[cat] = res.loc[3].tolist()
			precision_std[cat] = res.loc[4].tolist()
			recall[cat] = np.array(res.columns).tolist()

	result2 = {'name': scores['name'],
				'precision': [precision[i] for i in precision],
				'precision_std': [precision_std[i] for i in precision_std],
				'recall': [recall[i] for i in recall],
				'ap': scores['test_average_precision'], 
				'ap_std': scores['test_average_precision_std']}
	return result2

def quick_print(cars):
	for x in cars:
	    print('{}:{}'.format(x,cars[x]))

lines = {0:[], 1:[], 2:[], 3:[], 4:[]}
labels = {0:[], 1:[], 2:[], 3:[], 4:[]}

def run():

	X = np.load('{}/data.npy'.format(args.data))
	y = np.load('{}/label.npy'.format(args.data))
	X_flattened = X.reshape(X.shape[0], -1)

	result_arr = []

	cv = args.cv

	rf = RandomForestClassifier(class_weight='balanced')
	rf_cv_results = cross_validate(rf, X_flattened, y, cv, 'RandomForest')
	result_arr.append(rf_cv_results)

	rando = DummyClassifier(strategy="uniform")
	rando_cv_results = cross_validate(rando, X_flattened, y, cv, 'RandomClassifier')
	result_arr.append(rando_cv_results)

	p = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(50,), batch_size=64)
	p_cv_results = cross_validate(p, X_flattened, y, cv, 'SingleLayerPerceptron')
	result_arr.append(p_cv_results)
	print("Perceptron layers and iter: {}, {}".format(p.n_layers_, p.n_iter_))

	types = ['RandomForest', 'RandomClassifier', 'SingleLayerPerceptron']


	# scoring_types = ['accuracy', 'balanced_accuracy', 'f1', 'average_precision']
	for i, result in enumerate(result_arr):
		a_file = open(os.path.join(args.savepath, "{}.json".format(result['name'])), "w")
		json.dump(result, a_file)
		a_file.close()


if __name__== '__main__':
	run()