import Orange
import logging
import random
from discretization import *
from utils import *
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# Vars
validationsetPercentage = .3

# Utilities
logging.basicConfig(filename='main.log',level=logging.DEBUG,format='%(levelname)s\t%(message)s')

def logmessage(message, color):
	print color(message)
	logging.info(message)
	
def copyDataset(dataset):
	return Orange.data.Table(dataset)

# ============================================================================ #
boxmessage("Starting Phase 2: Working Data Splitting", warning)
workingSet = Orange.data.Table("workingset.tab")
workingSet.randomGenerator = Orange.orange.RandomGenerator(random.randint(0, 100))
workingSet.shuffle()
logmessage("Working Dataset Loaded", success)

# ============================================================================ #
# Split
info("Splitting Working Dataset for training and validation (70-30)")
selection = Orange.orange.MakeRandomIndices2(workingSet, validationsetPercentage)
	
validationSet = workingSet.select(selection, 0)
validationSet.save("step2_validationset.tab")

trainingSet = workingSet.select(selection, 1)
trainingSet.save("step2_trainingset.tab")

logmessage("Workset split performed", success)
logmessage("Training Instances: %s" % len(trainingSet), info)
logmessage("Validation Instances: %s" % len(validationSet), info)
# ============================================================================ #
