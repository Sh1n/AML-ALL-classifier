import Orange
import logging
import random
from utils import *
# Utilities
logging.basicConfig(filename='main.log',level=logging.DEBUG,format='%(levelname)s\t%(message)s')

def logmessage(message, color):
	print color(message)
	logging.info(message)
	
def copyDataset(dataset):
	return Orange.data.Table(dataset)

def pipeline():
	#execfile('step1.py') # Initial Split
	execfile('step2.py') # Working Set Split
	execfile('step3.py') # Discretization
	execfile('step4.py') # Feature Selection
	execfile('step5.py') # Model Training
	#execfile('step6.py') # Final Test


class pipelineThread (threading.Thread):
	def __init__(self, threadID):
		threading.Thread.__init__(self)
		self.threadID = threadID
	def run(self):
		pipeline()

# Compute S Threshold

computation = True
iterations = 10
multiThread = False
if computation:
	if not multiThread:
		for i in xrange(iterations):
			pipeline()
	else:
		threadID = 1
		threads = []
		for i in xrange(iterations):
			thread = pipelineThread(threadID)
			thread.start()
			threads.append(thread)
			threadID += 1
		for t in threads:
			t.join()
	
# Scores ===================================================================== #
scores = open('scores','r')
F1=[]
for line in scores:
	F1.append(float(line))
scores.close()
avg = sum(F1)/len(F1)
logmessage("Average F1: %s" % avg , info)

