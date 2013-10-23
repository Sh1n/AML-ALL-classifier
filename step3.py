import Orange
import logging
import random
from Discretizer import *
from FeatureSelector import *
from Normalizer import *
from utils import *
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# Vars
progress = False
K = 1
previousStep = 2
# Utilities
logging.basicConfig(filename='main.log',level=logging.DEBUG,format='%(levelname)s\t%(message)s')

def logmessage(message, color):
	print color(message)
	logging.info(message)
	
def copyDataset(dataset):
	return Orange.data.Table(dataset)


# Compute S Threshold

# ============================================================================ #
boxmessage("Starting Phase 3: Data Discretization", warning)
trainingSet = Orange.data.Table("step%s_trainingset.tab" % previousStep)
validationSet = Orange.data.Table("step%s_validationset.tab" % previousStep)
trainingSet.randomGenerator = Orange.orange.RandomGenerator(random.randint(0, 10))
logmessage("Training Dataset Loaded", success)

# ============================================================================ #

# Normalization
nz = Normalizer()
trainingSet = nz.normalize(trainingSet)
logmessage("Normalization applied", info)


# Discretization
ds = Discretizer(trainingSet, K, logging)
if progress:
	ds.load()
else:
	ds.findThresholds()
discretizedSet = ds.discretizeDataset(trainingSet)
discretizedSet.save("step3_trainingset.tab")

logmessage("Training Dataset Discretized", success)

discretizedValidationSet = ds.discretizeDataset(validationSet)
print len(discretizedValidationSet[0])
discretizedValidationSet.save("step3_validationset.tab")
logmessage("Validation Dataset Discretized", success)

# ============================================================================ #

