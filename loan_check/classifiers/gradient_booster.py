from pyspark.ml.classification import GBTClassifier

from .base_classifier import BaseClassifier

class GradientBoosterModel(BaseClassifier):
    def build_estimator(self):
        return GBTClassifier(
            featuresCol="features",
            labelCol="target",
            weightCol="weight",
            maxIter=50,
        )