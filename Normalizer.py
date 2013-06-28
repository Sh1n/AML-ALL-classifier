
class Normalizer:

	def normalize(self, dataset):
		for d in dataset:
			for f in dataset.domain:
				if not f == dataset.domain.class_var:
					d[f] = d[f] / sum(d)
		return dataset
