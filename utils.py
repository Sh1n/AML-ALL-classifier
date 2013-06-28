import threading
import Orange

class Threshold:
	def __init__(self, domain, threshold, infoGain, k):
		self.domain = domain
		self.threshold = threshold
		self.infogain = infoGain
		self.bestK = k

def copyDataset(dataset):
	return Orange.data.Table(dataset)

class discretizingThread (threading.Thread):
	def __init__(self, threadID, discretizer, feature, data):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.feature = feature
		self.data = data
		self.discretizer = discretizer
	def run(self):
		bestThreshold, bestInfogain = self.discretizer.discretizeAttribute(self.data, self.feature)
		self.discretizer.saveGain(self.feature, bestThreshold, bestInfogain)

def success(message):
	return "\033[92m%s\033[0m" % message

def info(message):
	return "\033[94m%s\033[0m" % message

def error(message):
	return "\033[91m%s\033[0m" % message

def warning(message):
	return "\033[93m%s\033[0m" % message

def boxmessage(message, color):
	print "============================"
	print "||%s" % color(message)
	print "============================"
