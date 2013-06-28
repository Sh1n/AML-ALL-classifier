import Orange
import logging
import random
import numpy as np
from utils import *
# Vars
testsetPercentage = .25

# Utilities
logging.basicConfig(filename='main.log',level=logging.DEBUG,format='%(levelname)s\t%(message)s')

def logmessage(message, color):
	print color(message)
	logging.info(message)
	
def copyDataset(dataset):
	return Orange.data.Table(dataset)


# Compute S Threshold

# =============================================================================
boxmessage("Starting Phase 1: Working & Testing split", warning)
data = Orange.data.Table("dataset.tab")
data.randomGenerator = Orange.orange.RandomGenerator(random.randint(0, 100))
logmessage("Main Dataset Loaded", success)

# =============================================================================
# Extracts Test Set
boxmessage("Extracting Test Set and Working Set", info)
testSet = None
workingSet = None

selection = Orange.orange.MakeRandomIndices2(data, testsetPercentage)

testSet = data.select(selection, 0)
testSet.save("finaltestset.tab")

workingSet = data.select(selection, 1)
workingSet.save("workingset.tab")

print success("Extraction performed")
print info("Test Instances: %s" % len(testSet))
print info("Work Instances: %s" % len(workingSet))
# =============================================================================
