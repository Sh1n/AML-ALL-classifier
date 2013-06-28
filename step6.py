import Orange
import logging
import random
from sklearn.externals import joblib
from discretization import *
from FeatureSelector import *
from utils import *
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn import preprocessing

# Vars
baseline = .9496
K = 1
S = 2000
C = 1.5
load = False
negate = False
# Utilities
logging.basicConfig(filename='main.log',level=logging.DEBUG,format='%(levelname)s\t%(message)s')

def logmessage(message, color):
	print color(message)
	logging.info(message)
	
def copyDataset(dataset):
	return Orange.data.Table(dataset)

# ============================================================================ #
boxmessage("Starting Phase 6: Final", warning)
testSet = Orange.data.Table("finaltestset.tab")
logmessage("Final Test Set loaded", info)

# Discretizer
ds = Discretizer(testSet, K, logging)
ds.load()
logmessage("Discretizer Loaded", info)

# Feature Selector
fs = FeatureSelector()
fs.load()
fs.setThreshold(S)
logmessage("Feature Selector Loaded", info)

# LabelEncoder
le = None
with open("labelencoder", "r") as in_file:
	le = pickle.load(in_file)
logmessage("Label Encoder Loaded", info)

# Model
clf = joblib.load('classifier.model')
logmessage("Classifier Loaded", info)

#discretizedSet = ds.discretizeDataset(trainingSet)

# ============================================================================ #

if not load:
	testSet3 = ds.discretizeDataset(testSet) # Apply Discretization
	logmessage("TestSet Discretized", success)
	testSet3.save("final_testset_discretized.tab")

if not load:
	testSet4 = fs.select(testSet3)
	logmessage("Feature Selection Applied", success)
	testSet4.save("final_testset_selected.tab")
else:
	testSet4 = Orange.data.Table("final_testset_selected.tab")

converted_test_data = ([le.transform([ d[f].value for f in testSet4.domain if f != testSet4.domain.class_var]) for d in testSet4])
converted_test_targets = le.transform([d[testSet4.domain.class_var].value for d in testSet4 ])
logmessage("Label Encoding Applied", success)

print converted_test_targets

logmessage("Starting Prediction Task", info)
prediction = clf.predict(converted_test_data)
if negate:
	prediction = np.array([933 if p == 934 else 934 for p in prediction])

print "Prediction: \t", prediction

print classification_report(converted_test_targets, prediction)

p, r, f1, support = precision_recall_fscore_support(converted_test_targets, prediction,average="micro") 

print p
print r
print f1
print support

# Save scores
#scores = open('scores', 'a')
#scores.write("%s\n" % (np.average(f1)))
#scores.close()

# Save scores
scores = open('scores', 'a')
scores.write("%s\n" % (np.average(f1)))
scores.close()

f1_avg = np.average(f1)

logmessage("Average F1(Over 2 classes): %s" % f1_avg, info)
if f1_avg > baseline:
       logmessage("Performance Increased", success)
else:
       logmessage("Performance Decreased", error)



#le = preprocessing.LabelEncoder()
#le.fit() # All the possible labels for all classes

# =========================== #
# Encode all labels
#labels = [d[f].value for f in trainingSet.domain for d in trainingSet] + [d[f].value for f in validationSet.domain for d in validationSet]
#le.fit(labels)
# =========================== #


# Convert Train Dataset
# Apply transformation, from labels to you know what I mean
#converted_train_data = ([le.transform([ d[f].value for f in trainingSet.domain if f != trainingSet.domain.class_var]) for d in trainingSet])

#converted_train_data = [dict(enumerate(d)) for d in converted_train_data]
#converted_train_data = vector.fit_transform(converted_train_data)

#logmessage("Validation dataset is %s" % len(validationSet), info)
#logmessage("Validation dataset features are %s" % len(validationSet.domain), info)

#print len(converted_train_data)
#print len(converted_train_data[0])

#converted_train_targets = le.transform([d[trainingSet.domain.class_var].value for d in trainingSet ])

#clf = svm.SVC(kernel='linear',C=C, class_weight="auto")
#clf.fit(converted_train_data, converted_train_targets)
#logmessage("Model learnt", success)

# Performances


# Convert Test Dataset
#converted_test_data = ([le.transform([ d[f].value for f in validationSet.domain if f != validationSet.domain.class_var]) for d in validationSet])
#converted_test_data = [dict(enumerate(d)) for d in converted_test_data]
#converted_test_data = vector.fit_transform(converted_test_data)


#print len(converted_test_data)
#print len(converted_test_data[0])

#converted_test_targets = le.transform([d[validationSet.domain.class_var].value for d in validationSet ])

#logmessage("Starting Prediction Task", info)
#prediction = clf.predict(converted_test_data)
#p, r, f1, support = precision_recall_fscore_support(converted_test_targets, prediction) 

# Save scores
#scores = open('scores', 'a')
#scores.write("%s\n" % (np.average(f1)))
#scores.close()
#f1_avg = np.average(f1)

#logmessage("Average F1(Over 2 classes): %s" % f1_avg, info)
#if f1_avg > baseline:
#	logmessage("Performance Increased", success)
#else:
#	logmessage("Performance Decreased", error)
 	
#logmessage("Saving Model", info)
#joblib.dump(clf, 'classifier.model')
# =============================================================================
