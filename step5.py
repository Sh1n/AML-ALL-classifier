import Orange
import logging
import random
from sklearn.externals import joblib
from utils import *
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn import preprocessing

# Vars
testsetPercentage = .2
validationsetPercentage = .3
progress = False
baseline = .9496
C = 0.5
kernel = 'poly'
previousStep = 4
# Utilities
logging.basicConfig(filename='main.log',level=logging.DEBUG,format='%(levelname)s\t%(message)s')

def logmessage(message, color):
	print color(message)
	logging.info(message)
	
def copyDataset(dataset):
	return Orange.data.Table(dataset)

# Compute S Threshold

# ============================================================================ #
boxmessage("Starting Phase 5: Model Learning", warning)
trainingSet = Orange.data.Table("step%s_trainingset.tab" % previousStep)
validationSet = Orange.data.Table("step%s_validationset.tab" % previousStep)
trainingSet.randomGenerator = Orange.orange.RandomGenerator(random.randint(0, 10))
logmessage("Feature Selected Working Dataset Loaded", success)

logmessage("Training dataset is %s" % len(trainingSet), info)
logmessage("Training dataset features are %s" % len(trainingSet.domain), info)

logmessage("Validation dataset is %s" % len(validationSet), info)
logmessage("Validation dataset features are %s" % len(validationSet.domain), info)

# ============================================================================ #

le = preprocessing.LabelEncoder()
#le.fit() # All the possible labels for all classes

# =========================== #
# Encode all labels
labels = [d[f].value for f in trainingSet.domain for d in trainingSet] + [d[f].value for f in validationSet.domain for d in validationSet] + ['?']
le.fit(labels)
with open("labelencoder", "wb") as out_file:
	pickle.dump(le, out_file)
# =========================== #


# Convert Train Dataset
# Apply transformation, from labels to you know what I mean
converted_train_data = ([le.transform([ d[f].value for f in trainingSet.domain if f != trainingSet.domain.class_var]) for d in trainingSet])

# Weights
ALL = trainingSet.select(gene='ALL')
AML = trainingSet.select(gene='AML')

#converted_train_data = [dict(enumerate(d)) for d in converted_train_data]
#converted_train_data = vector.fit_transform(converted_train_data)

logmessage("Validation dataset is %s" % len(validationSet), info)
logmessage("Validation dataset features are %s" % len(validationSet.domain), info)

print len(converted_train_data)
print len(converted_train_data[0])

converted_train_targets = le.transform([d[trainingSet.domain.class_var].value for d in trainingSet ])
print converted_train_targets

clf = svm.SVC(kernel=kernel,C=C)
clf.fit(converted_train_data, converted_train_targets)
logmessage("Model learnt", success)

# Performances


# Convert Test Dataset
converted_test_data = ([le.transform([ d[f].value for f in validationSet.domain if f != validationSet.domain.class_var]) for d in validationSet])
#converted_test_data = [dict(enumerate(d)) for d in converted_test_data]
#converted_test_data = vector.fit_transform(converted_test_data)


print len(converted_test_data)
print len(converted_test_data[0])

converted_test_targets = le.transform([d[validationSet.domain.class_var].value for d in validationSet ])

logmessage("Starting Prediction Task", info)
prediction = clf.predict(converted_test_data)
print "Predicted \t", prediction
print "Truth \t",converted_test_targets
p, r, f1, support = precision_recall_fscore_support(converted_test_targets, prediction) 

# Save scores
#scores = open('scores', 'a')
#scores.write("%s\n" % (np.average(f1)))
#scores.close()
f1_avg = np.average(f1)

logmessage("Average F1(Over 2 classes): %s" % f1_avg, info)
if f1_avg > baseline:
	logmessage("Performance Increased", success)
else:
	logmessage("Performance Decreased", error)
 	
logmessage("Saving Model", info)
joblib.dump(clf, 'classifier.model')
# =============================================================================
