import Orange
from Orange.classification import svm
from Orange.evaluation import testing, scoring
data = Orange.data.Table("dataset.tab")
learner = svm.SVMLearner(verbose=True, normalization= True)
results = testing.cross_validation([learner], data, folds = 1)
print "CA:  %.4f" % scoring.CA(results)[0]
print "AUC: %.4f" % scoring.AUC(results)
