from __future__ import division
from sklearn import svm
import Orange
from sklearn import cross_validation
from sklearn.metrics import f1_score, precision_recall_fscore_support

import numpy as np
import random

orangeData = Orange.data.Table("dataset.tab")
F1 = []
PRE = []
REC = []
K = 10
oneShot = True
orangeData.randomGenerator = Orange.orange.RandomGenerator(random.randint(0, 10))
folds = Orange.orange.MakeRandomIndicesCV(orangeData, K)
labels = ['ALL', 'AML']

for i in xrange(K):

	testDataset = orangeData.select(folds, i)
	trainDataset = orangeData.select(folds, i, negate = 1)
	
	# Convert Train Dataset
	converted_train_data = np.array([[ d[f].value for f in orangeData.domain if f != orangeData.domain.class_var] for d in trainDataset])
	converted_train_targets = np.array([ 0 if d[orangeData.domain.class_var].value == 'ALL' else 1 for d in trainDataset ])

	# Train SVM
	clf = svm.SVC(kernel='linear')
	clf.fit(converted_train_data, converted_train_targets)

	# Testing

	# Convert Test Dataset
	converted_test_data = np.array([[ d[f].value for f in orangeData.domain if f !=	orangeData.domain.class_var] for d in testDataset])
	converted_test_targets = np.array([0 if d[orangeData.domain.class_var].value == 'ALL' else 1 for d in testDataset ])

	if oneShot:
		prediction = clf.predict(converted_test_data)
		p, r, f1, support = precision_recall_fscore_support(converted_test_targets, prediction, average='micro') 
		F1.append(np.average(f1))
		REC.append(r)
		PRE.append(p)
		# One Shot Mode
	else:
		# Confusion Matrix
		cfMatrix = {"ALL":{ "ALL": 0, "AML": 0 }, "AML":{ "ALL": 0, "AML": 0}}

		for j in xrange(len(converted_test_data)):
			prediction = clf.predict(converted_test_data[j])
		#print "Predicted %s, truth is %s" % (prediction, converted_test_targets[i])
			cfMatrix[converted_test_targets[j].value][prediction[0].value] += 1

		PRE.append(cfMatrix['ALL']['ALL'] / (cfMatrix['ALL']['ALL'] + cfMatrix['AML']['ALL']))
		REC.append(cfMatrix['ALL']['ALL'] / (cfMatrix['ALL']['ALL'] + cfMatrix['ALL']['AML']))
		F1.append((2*(REC[i] * PRE[i])) / (REC[i] + PRE[i]))
	
# Computing Score
Fm = (1/K) * (sum(F1))
print "Average F1 Score ", Fm

Recm = (1/K) * sum(REC)
print "Average Recall ", Recm

Prem = (1/K) * sum(PRE)
print "Average Precision ", Prem

f = open('baseline.result', 'w')
f.write('AVG_F1\t')
f.write(str(Fm))
f.write('\n')
f.write('AVG_REC\t')
f.write(str(Recm))
f.write('\n')
f.write('AVG_PRE\t')
f.write(str(Prem))
f.write('\n')


