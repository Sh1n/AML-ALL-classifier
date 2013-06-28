import operator
import itertools
import Orange
from utils import *
import math
import pickle
import numpy as np

class FeatureSelector:
	# methods are best-s, threshold, covariance
	def __init__(self):
		self.method = "best-s"
		self.threshold = .0
		self.avg = 0
		self.var = 0
		pass
	
	def computeThreshold(self, dataset):
		if self.method == "covariance":
			self.avg = []
			# Compute the Mu for each features
			for f in dataset.domain:
				if not f == dataset.domain.class_var:
					self.avg.append((f,np.average([d[f].value for d in dataset])))
		else:
			self.gains = []
			gain = Orange.feature.scoring.InfoGain()
			for d in dataset.domain:
				if not d == dataset.domain.class_var:
					print info("Computing Infogain of %s" % d)
					self.gains.append((d, gain(d, dataset)))
			self.gains.sort(key=operator.itemgetter(1),reverse=True)
			if not self.method == "best-s":
				self.avg = self.average(self.gains)
				print info("Average is %s" % self.avg)
				self.var = self.variation(self.gains)
				print info("Variance is %s" % self.var)
				self.std = self.std_dev(self.gains)
				print info("Std.Deviation is %s" % self.std)
				print info("Using Average as Threshold")		
				self.threshold = math.fabs(self.avg - self.std)
				print success("Computed Threshold is %s" % self.threshold)

	def save(self):
		with open("threshold", "wb") as out_file:
			pickle.dump({"gains": self.gains, "avg": self.avg, "var": self.var}, out_file)

	def load(self):
		with open("threshold", "r") as in_file:
			temp = pickle.load(in_file)
			self.gains = temp['gains']
			self.avg = temp['avg']
			self.var = temp['var']

	def setThreshold(self, threshold):
		self.threshold = threshold

	def select(self, dataset):
		# order gains!!!
		features = []
		counter = 1
		for d,g in self.gains:
			if self.method == "best-s":
				features.append(d)
				counter += 1
				if counter >= self.threshold:
					break
			else:		
				if g > self.threshold:
					features.append(d)
		new_domain = Orange.data.Domain(features + [dataset.domain.class_var])
		return Orange.data.Table(new_domain, dataset)
		#data = copyDataset(dataset)
		#features = Orange.feature.selection.selectAttsAboveThresh(self.gains, self.threshold)
		#print len(features)		
		#return Orange.feature.selection.FilterAttsAboveThresh(data, threshold=self.threshold, measure = Orange.feature.scoring.InfoGain())
	

	# Average
	def average(self, gains):
		return np.mean([g for d,g in gains.items()])
	
	def variation(self, gains):
		return np.var([g for d,g in gains.items()])

	# Standard Deviation
	def std_dev(self, gains):
		return np.std([g for d,g in gains.items()])
		

	# Information gain computation procedures
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
		gain = Orange.feature.scoring.InfoGain()
		return gain(attribute, data)		
		"""
		val_frequencies = self.values_frequencies(data, attribute)
		subset_entropy = 0.0
	
	   	for value in val_frequencies.keys():
			value_prob = val_frequencies[value] / sum(val_frequencies.values())
		
			data_subset = data.filter({attribute:value})
			subset_entropy += value_prob * self.entropy(data_subset)
		
		return (self.entropy(data)) - subset_entropy
		"""

	
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

