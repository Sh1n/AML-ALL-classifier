import Orange
import math
import logging
import operator
import itertools
import cPickle
import threading
import time
# Changelog 11062013 1333
# Adding multithread support

class Threshold:
	def __init__(self, domain, threshold, infoGain, k):
		self.domain = domain
		self.threshold = threshold
		self.infogain = infoGain
		self.bestK = k

class discretizingThread (threading.Thread):
	def __init__(self, threadID, feature, data, K):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.feature = feature
		self.data = data
	def run(self):
		bestThreshold, bestInfogain = discretize(self.data, self.feature, K)
		saveGain(self.feature, bestThreshold, bestInfogain, K)
        
"""
	Performance Measurement
"""
def precision():
	# precision = (True Positive / (True Positive + False Positive))
	



logging.basicConfig(filename='discretizer.log',level=logging.DEBUG,format='%(levelname)s\t%(message)s')
"""
	Given in input an Orange Table, extracts it's entropy
"""
def entropy(data):
	# sommatoria, per i da 1 a c (tutti i valori possibili) di p1 log2 p1 , p1 = #elementi con valore i
	val_frequencies = values_frequencies(data, data.domain.class_var) # value -> freq
	entropy_value = 0.0
	for frequency in val_frequencies.values():
		entropy_value += (-frequency/len(data)) * math.log((frequency/len(data)), 2)
	return entropy_value	

def informationGain(data, attribute):
	val_frequencies = values_frequencies(data, attribute)
	subset_entropy = 0.0
	
   	for value in val_frequencies.keys():
		value_prob = val_frequencies[value] / sum(val_frequencies.values())
		
		data_subset = data.filter({attribute:value})#Orange.data.Table([instance for instance in data if instance[attribute].value == value] # convertire in tabella
		subset_entropy += value_prob * entropy(data_subset)
    
	return (entropy(data)) - subset_entropy


"""
	Given a table returns the frequencies of every distinct element
"""
def values_frequencies(data, attribute):
	values_freq = {}
	
	for instance in data:
		if values_freq.has_key(instance[attribute].value):
			values_freq[instance[attribute].value] += 1.0
		else:
			values_freq.update({instance[attribute].value : 1.0})
	return values_freq

def dispose(Set,Card):
	"""
		Retrieves all possible subsets of a given set of a given cardinality
	"""
	return set(itertools.combinations(Set, Card))


def splitDataset(dataset, attribute, values):
	"""
		Split in intervals defined by the values passed
		k values -> k + 1 intervals
	"""
	intervalDiscretizer = Orange.feature.discretization.IntervalDiscretizer(points = values)
	newDomain = intervalDiscretizer.construct_variable(attribute)
	discretizedDataset = data.select([newDomain, data.domain.classVar])
	return discretizedDataset


def green(message):
	return "\033[92m%s\033[0m" % message

def blue(message):
	return "\033[94m%s\033[0m" % message

def red(message):
	return "\033[91m%s\033[0m" % message

def yellow(message):
	return "\033[93m%s\033[0m" % message

def copyDataset(dataset):
	return Orange.data.Table(dataset)


"""
	Given a table of data discretizes it
"""
def discretize(data, attribute, K=1):
	logging.info("Discretizing: %s" % attribute)
	print "Discretizing: %s" % blue(attribute)
	# Copia del vecchio ?
	data = copyDataset(data)
	data.sort();
	thresholds = []
	for i in xrange(len(data) - 1):
		current = data[i]
		next = data[i+1]
		if current.get_class() != next.get_class():
			thresholds.append((next[attribute].value - current[attribute].value)/2)
	thresholds = list(set(thresholds))
	thresholds.sort()
	logging.debug("\tCandidate Thresholds: %s" % thresholds)
	gains = []
	for candidate in dispose(thresholds, K):
		splittedDataset = splitDataset(data, attribute, candidate)
		infogain = informationGain(splittedDataset, "D_%s" % attribute.name)
		logging.debug("\t\tWorking on candidate threshold %s with an infogain of %s" % (candidate, infogain))
		gains.append((candidate,infogain))

	gains.sort(key=operator.itemgetter(1),reverse=True) # DESC order
	bestThreshold, maxInfogain = gains[0]
	logging.info( "\tBest candidate is %s with an information gain of: %s" % (bestThreshold,maxInfogain))
	print "\tBest candidate is %s with an information gain of: %s" % (green(','.join([str(i) for i in bestThreshold])), green(maxInfogain))
	return bestThreshold , maxInfogain
	
print "============================"
print "||          %s         ||" % red("Start")
print "============================"

data = Orange.data.Table("dataset.tab")
m = len(data.domain.features)
m_cont = sum(1 for x in data.domain.features if x.var_type==Orange.data.Type.Continuous)
print "%d features, %d continuous" % (m, m_cont)
print "Class:", data.domain.class_var.name
print "Possible class values: ", data.domain.class_var.values


# Feature Discretization
K = 1	# not implemented yet <--- one of the variable.
i = 0

#In this way we do not copute the best K for each attribute. Hence k should be considered as a superior bound:
# the algorithm computes the infogain with a particoular K, once took, even if the value K is increaed, bu the 
# best threshold is still found on a lower K, we took it into consideration, simply considering that
# the evluation of a domain goes from k = 1 : K
# new type = threshold= bestValue, bestCandidate, bestk

# So according to this we must mantain a dictionary of 5147 objects with information
# about data clustering. At each variation of K
gains = {}
def saveGain(domain, bestThreshold, bestInfogain, K):
	global gains
	if domain not in gains.keys():
		gains.update({domain: Threshold(domain, bestThreshold, bestInfogain, K)})
	else:
		current = gains[domain]
		if bestInfogain > current.infogain:
				gains[domain]= Threshold(domain, bestThreshold, bestInfogain, K)

print "Starting: %s" % (time.ctime(time.time()))
for i in range(K):
	print "Feature Discretization with K = ", i + 1
	threads = []
	threadId = 1
	for d in data.domain:
		if not d == data.domain.class_var:
			# Discretize on every feature according to information gain:
			# We simply build a new dataset for each feature by bulding a new domain			
			new_domain = Orange.data.Domain([d] + [data.domain.class_var])
			new_data = Orange.data.Table(new_domain, data)
			thread = discretizingThread(threadId, d, new_data, i + 1)
			thread.start()
			threads.append(thread)			
			#bestThreshold, bestInfogain = discretize(new_data, d, i + 1)
			#saveGain(d, bestThreshold, bestInfogain, i + 1)
			# Here we should update the table (column d) according to the best threshold

	# Wait for all threads to complete
	for t in threads:
		t.join()
	print "Discretization finished on K = ", K


print "Savin Gains"	
out_file = open("gains.multithread.shin","w")
out_file.write(cPickle.dumps(gains))
out_file.close()
print "Ending: %s" % (time.ctime(time.time()))
