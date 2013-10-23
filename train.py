from __future__ import division
from sklearn import svm
import Orange
from sklearn import cross_validation
import numpy as np
import random

"""
	Version 2:
		K-fold with K = 7
"""
orangeData = Orange.data.Table("dataset.tab")
F1 = []
orangeData.randomGenerator = Orange.orange.RandomGenerator(random.randint(0, 10))
folds = Orange.orange.MakeRandomIndicesCV(orangeData, 7)

print folds
iterations = 10
for i in xrange(iterations):
	orangeData.shuffle()

	# Extract Test Dataset
	testDataset = Orange.preprocess.selectPRandom(orangeData, P=25)

	# Extract Train Dataset
	trainDataset = [d for d in orangeData if d not in testDataset]

	# Convert Train Dataset
	converted_train_data = np.array([[ d[f].value for f in orangeData.domain if f != orangeData.domain.class_var] for d in trainDataset])
	converted_train_targets = np.array([d[orangeData.domain.class_var] for d in trainDataset ])

	# Train SVM
	clf = svm.SVC(kernel='linear')
	clf.fit(converted_train_data, converted_train_targets)

	# Testing

	# Convert Test Dataset
	converted_test_data = [[ d[f].value for f in orangeData.domain if f !=	orangeData.domain.class_var] for d in testDataset]
	converted_test_targets = [d[orangeData.domain.class_var] for d in testDataset ]

	# Confusion Matrix
	"""
	   |ALL|AML			PREDICTED	 
	ALL|   |		  TR
	AML|   |          UE
	"""
	cfMatrix = {"ALL":{ "ALL": 0, "AML": 0 }, "AML":{ "ALL": 0, "AML": 0}}

	for j in xrange(len(converted_test_data)):
		prediction = clf.predict(converted_test_data[j])
	#print "Predicted %s, truth is %s" % (prediction, converted_test_targets[i])
		cfMatrix[converted_test_targets[j].value][prediction[0].value] += 1

	# Precision
	PRE = cfMatrix['ALL']['ALL'] / (cfMatrix['ALL']['ALL'] + cfMatrix['AML']['ALL'])
	REC = cfMatrix['ALL']['ALL'] / (cfMatrix['ALL']['ALL'] + cfMatrix['AML']['AML'])
	print "Precision ", PRE
	print "Recall ", REC
	F1.append((2*(REC * PRE)) / (REC + PRE))
	print "F1 ", (2*(REC * PRE)) / (REC + PRE)
	"""
		ALL Statistics
	
	PRE_ALL = cfMatrix['ALL']['ALL'] / (cfMatrix['ALL']['ALL'] + cfMatrix['AML']['ALL'])
	REC_ALL = cfMatrix['ALL']['ALL'] / (cfMatrix['ALL']['ALL'] + cfMatrix['ALL']['AML'])

	print "ALL pre %0.4f, rec %0.4f" % (PRE_ALL, REC_ALL) 


		AML Statistics
	
	PRE_AML = cfMatrix['AML']['AML'] / (cfMatrix['AML']['AML'] + cfMatrix['ALL']['AML'])
	REC_AML = cfMatrix['AML']['AML'] / (cfMatrix['AML']['AML'] + cfMatrix['AML']['ALL'])

	print "AML pre %0.4f, rec %0.4f" % (PRE_AML, REC_AML) 

	
		MultiClass Accouracy
	
	M_ACC = cfMatrix['ALL']['ALL'] + cfMatrix['AML']['AML'] / (cfMatrix['AML']['AML'] + cfMatrix['AML']['ALL'] + cfMatrix['ALL']['AML'] + cfMatrix['ALL']['ALL'])
	print "Accouracy ", M_ACC

	PREm = cfMatrix['ALL']['ALL'] + cfMatrix['AML']['AML'] / (cfMatrix['ALL']['ALL'] + cfMatrix['AML']['AML'] + cfMatrix['AML']['ALL'] + cfMatrix['ALL']['AML'])

	RECm = cfMatrix['ALL']['ALL'] + cfMatrix['AML']['AML'] / (cfMatrix['ALL']['ALL'] + cfMatrix['AML']['AML'] + cfMatrix['ALL']['AML'] + cfMatrix['AML']['ALL'])

	print "RecM %0.4f, PreM %0.4f" % (RECm, PREm) 

	F1.append((2*(RECm * PREm)) / (RECm + PREm))
	print "F1 measure ", F1
	print "===================================="
	"""
f = open('baseline', 'w')
f.write('\n'.join(map(str, F1)))
f.close()

"""
# Precision
PRE_ALL = ALL_P / (ALL_P + ALL_FP)
PRE_AML = AML_P / (AML_P + AML_FP)

print "Precision ALL ", PRE_ALL
print "Precision AML ", PRE_AML

# Recall

 
print "ALL Predicted Correclty ", ALL_P
print "AML Predicted Correctly ", AML_P
print "Wrong ALL %s, AML %s" % (ALL_FP, AML_FP)
"""
