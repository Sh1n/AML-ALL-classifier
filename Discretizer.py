import threading
import Orange
import cPickle
import math
import operator
import itertools
import pickle
import cPickle
from utils import *

class Discretizer:

	def __init__(self, data, K, logging):
		self.data = data
		self.gains = {}
		self.K = K
		self.logger = logging

	def load(self):
		with open("discretizer.K.gains","rb") as in_file:
			self.gains = pickle.load(in_file)
	
	def save(self):
		# Save Gains
		with open("discretizer.K.gains","w") as out_file:
			pickle.dump(self.gains, out_file)

	def getGains(self):
		return self.gains
	
	def discretizeDataset(self, data):
		"""
			Returns the new dataset according to gains found
		"""
		print "Starting Discretization"
		data = copyDataset(data)
		for gain, threshold in self.gains.items():
			intervalDiscretizer = Orange.feature.discretization.IntervalDiscretizer(points = threshold.threshold)
			newDomain = intervalDiscretizer.construct_variable(gain)
			data = data.select([newDomain] + [d for d in data.domain if not d == gain])
		print "Discretization Ended"
		return data

	# Starts Discretization Procedure, multithreaded
	def findThresholds(self):
		self.gains = {}
		threads = []
		threadID = 1
		for d in self.data.domain:
			if not d == self.data.domain.class_var:
				new_domain = Orange.data.Domain([d] + [self.data.domain.class_var])
				new_data = Orange.data.Table(new_domain, self.data)
				thread = discretizingThread(threadID, self, d, new_data)
				thread.start()
				threads.append(thread)
				threadID += 1
		for t in threads:
			t.join()
		print success("Discretization finished")
		print info("saving results")
		self.save();
	
	def dispose(self, Set, Card):
		"""
			Retrieves all possible subsets of a given set of a given cardinality
		"""
		return set(itertools.combinations(Set, Card))

	def discretizeAttribute(self, data, attribute):
		"""
			Discretizes the given attribute with the given K
		"""
		self.logger.info("Discretizing: %s" % attribute)
		print "Discretizing: %s" % info(attribute)
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
		self.logger.debug("Candidate Thresholds: %s" % thresholds)
		localgains = []
		gainScorer = Orange.feature.scoring.InfoGain()
		for candidate in self.dispose(thresholds, self.K):
			splittedDataset = self.splitDataset(data, attribute, candidate)
			infogain = gainScorer( "D_%s" % attribute.name, splittedDataset)
			self.logger.debug("Working on candidate threshold %s with an infogain of %s" % (candidate, infogain))
			localgains.append((candidate,infogain))

		localgains.sort(key=operator.itemgetter(1),reverse=True) # DESC order
		bestThreshold, maxInfogain = localgains[0]
		self.logger.info( "\tBest candidate is %s with an information gain of: %s" % (bestThreshold,maxInfogain))
		print "\tBest candidate is %s with an information gain of: %s" % (success(','.join([str(i) for i in bestThreshold])), success(maxInfogain))
		return bestThreshold , maxInfogain
	

	# Save Gains
	def saveGain(self, domain, bestThreshold, bestInfogain):
		if domain not in self.gains.keys():
			self.gains.update({domain: Threshold(domain, bestThreshold, bestInfogain, self.K)})
		else:
			current = self.gains[domain]
			if bestInfogain > current.infogain:
				self.gains[domain]= Threshold(domain, bestThreshold, bestInfogain, self.K)

	def entropy(self, data):
		"""
			Given in input an Orange Table, extracts it's entropy
		"""
		val_frequencies = self.values_frequencies(data, data.domain.class_var) # value -> freq
		entropy_value = 0.0
		for frequency in val_frequencies.values():
			entropy_value += (-frequency/len(data)) * math.log((frequency/len(data)), 2)
		return entropy_value	

	def informationGain(self, data, attribute):
		val_frequencies = self.values_frequencies(data, attribute)
		subset_entropy = 0.0
	
	   	for value in val_frequencies.keys():
			value_prob = val_frequencies[value] / sum(val_frequencies.values())
		
			data_subset = data.filter({attribute:value})
			subset_entropy += value_prob * self.entropy(data_subset)
		
		return (self.entropy(data)) - subset_entropy

	
	def values_frequencies(self, data, attribute):
		"""
			Given a table returns the frequencies of every distinct element
		"""
		values_freq = {}
	
		for instance in data:
			if values_freq.has_key(instance[attribute].value):
				values_freq[instance[attribute].value] += 1.0
			else:
				values_freq.update({instance[attribute].value : 1.0})
		return values_freq

	def splitDataset(self, data, attribute, values):
		"""
			Split in intervals defined by the values passed
			k values -> k + 1 intervals
		"""
		intervalDiscretizer = Orange.feature.discretization.IntervalDiscretizer(points = values)
		newDomain = intervalDiscretizer.construct_variable(attribute)
		discretizedDataset = data.select([newDomain, data.domain.classVar])
		return discretizedDataset


