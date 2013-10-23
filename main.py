import Orange
import logging
import random
from discretization import *
from FeatureSelector import *
from utils import *
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# Vars
testsetPercentage = .2
validationsetPercentage = .3
progress = False
baseline = .9496

# Utilities
logging.basicConfig(filename='main.log',level=logging.DEBUG,format='%(levelname)s\t%(message)s')

def logmessage(message, color):
	print color(message)
	logging.info(message)
	
def copyDataset(dataset):
	return Orange.data.Table(dataset)


# Compute S Threshold

# =============================================================================
boxmessage("Start", warning)
data = Orange.data.Table("dataset.tab")
data.randomGenerator = Orange.orange.RandomGenerator(random.randint(0, 10))
logmessage("Main Dataset Loaded", success)

# =============================================================================
# Extracts Test Set
boxmessage("Extracting Test Set and Working Set", info)
testSet = None
workingSet = None
if progress:
	try:
		with open("finaltestset.tab"):
			logmessage("Final Test Set found", info)
		with open("trainingset.tab"):
			logmessage("Working Set found", info)
		testSet = Orange.data.Table("finaltestset.tab")
		workingSet = Orange.data.Table("trainingset.tab")
	except IOError:
		logmessage("IOError in loading final and working sets", error)
		pass
else:
	selection = Orange.orange.MakeRandomIndices2(data, testsetPercentage)
	testSet = data.select(selection, 0)
	testSet.save("finaltestset.tab")
	workingSet = data.select(selection, 1)
	workingSet.save("workingset.tab")

print success("Extraction performed")
print info("Test Instances: %s" % len(testSet))
print info("Training + Validation Instances: %s" % len(workingSet))
# =============================================================================
# Starts Iterations
K = 1
S = 0
C = 0
boxmessage("Starting main Loop", info)
#while(performanceIncrease):

# Split
if not progress:
	info("Splitting Working Dataset for training and validation (70-30)")
	selection = Orange.orange.MakeRandomIndices2(workingSet, validationsetPercentage)
	validationSet = workingSet.select(selection, 0)
	trainingSet = workingSet.select(selection, 1)
	trainingSet.save("trainingset.tab")
	validationSet.save("validationset.tab")
else:
	validationSet = Orange.data.Table("validationset.tab")
	trainingSet = Orange.data.Table("trainingset.tab")

# Discretization
ds = Discretizer(trainingSet, K, logging)
if progress:
	try:
		with open("discretizer.K.gains"): 
			print info("Loading Previous Iteration")
			ds.load()
	except IOError:
		logmessage("IOError in loading found gains", error)
		pass
else:
	ds.findThresholds()
   
if progress:
	try:
		with open("discretized.tab"):
			trainingSet = Orange.data.Table("discretized.tab")
			print info("Discretized Dataset Loaded")
	except IOError:
		logmessage("IOError in loading discretized training dataset", error)
else:
	trainingSet = ds.discretizeDataset(trainingSet)
	trainingSet.save("discretized.tab")

# ============================================================================ #
# Feature Selection
fs = FeatureSelector()
if progress:
	try:
		with open("featureselected.tab"):
			trainingSet = Orange.data.Table("featureselected.tab")
			print info("Features Selected Dataset Loaded")
	except IOError:
		fs.computeThreshold(trainingSet)
		fs.save()
		trainingSet = fs.select(trainingSet)
		trainingSet.save("featureselected.tab")

print info("New training dataset is %s" %len(trainingSet))
print info("New training dataset features are %s" % len(trainingSet[0]))
	
# Model Training

# Convert Train Dataset
# Apply transformation, from labels to you know what I mean
converted_train_data = ([[ d[f].value for f in trainingSet.domain if f != trainingSet.domain.class_var] for d in trainingSet])
converted_train_data = [dict(enumerate(d)) for d in converted_train_data]
vector = DictVectorizer(sparse=False)
converted_train_data = vector.fit_transform(converted_train_data)

converted_train_targets = ([ 0 if d[trainingSet.domain.class_var].value == 'ALL' else 1 for d in trainingSet ])

clf = svm.SVC(kernel='linear')
clf.fit(converted_train_data, converted_train_targets)
logmessage("Model learnt", success)

# Performances

# Apply Discretization and feature selection to validation set
validationSet = ds.discretizeDataset(validationSet)
validationSet = fs.select(validationSet)
logmessage("Validation set length is %s" % len(validationSet), info)
logmessage("Validation feature length is %s" % len(validationSet[0]), info)
	# Convert Test Dataset
converted_test_data = ([[ d[f].value for f in validationSet.domain if f !=	validationSet.domain.class_var] for d in validationSet])
converted_test_data = [dict(enumerate(d)) for d in converted_test_data]
converted_test_data = vector.fit_transform(converted_test_data)

converted_test_targets = ([0 if d[validationSet.domain.class_var].value == 'ALL' else 1 for d in validationSet ])

logmessage("Starting Prediction Task", info)
prediction = clf.predict(converted_test_data)
p, r, f1, support = precision_recall_fscore_support(converted_test_targets, prediction) 
f1_avg = np.average(f1)
logmessage("Average F1(Over 2 classes): %s" % f1_avg, info)
if f1_avg > baseline:
	logmessage("Performance Increased", success)
	logmessage("Using K: %s, S: %s, C: default" % (ds.K, fs.threshold), info)
else:
	logmessage("Performance Decreased", error)
 	
# =============================================================================
# Final Test
