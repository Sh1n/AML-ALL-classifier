import Orange
import logging
import random
from FeatureSelector import *
from utils import *
import numpy as np

# Vars
baseline = .9496
S = 2000 # top-s features
progress = False
previousStep = 3

# Utilities
logging.basicConfig(filename='main.log',level=logging.DEBUG,format='%(levelname)s\t%(message)s')

def logmessage(message, color):
	print color(message)
	logging.info(message)
	
def copyDataset(dataset):
	return Orange.data.Table(dataset)


# Compute S Threshold

# ============================================================================ #
boxmessage("Starting Phase 4: Feature Selection", warning)
trainingSet = Orange.data.Table("step%s_trainingset.tab" % previousStep)
validationSet = Orange.data.Table("step%s_validationset.tab" % previousStep)
trainingSet.randomGenerator = Orange.orange.RandomGenerator(random.randint(0, 10))
logmessage("Discretized Working Dataset Loaded", success)

# ============================================================================ #
# Feature Selection
fs = FeatureSelector()
if progress:
	fs.load()
else:
	fs.computeThreshold(trainingSet)
fs.setThreshold(S)

fs.save()
selectedtrainingSet = fs.select(trainingSet)
selectedtrainingSet.save("step4_trainingset.tab")

logmessage("New training dataset is %s" % len(selectedtrainingSet), info)
logmessage("New training dataset features are %s" % len(selectedtrainingSet.domain), info)

selectedvalidationset = fs.select(validationSet)
selectedvalidationset.save("step4_validationset.tab")

logmessage("New training dataset is %s" % len(selectedvalidationset), info)
logmessage("New training dataset features are %s" % len(selectedvalidationset.domain), info)

if len(selectedvalidationset.domain) == len(selectedtrainingSet.domain):
	logmessage("Training and Validation sets have the same cardinality", success)
else:
	logmessage("Training and Validation sets do not have the same cardinality", error)

logmessage("Feature Selection Complete", success)
# ============================================================================ #

	
