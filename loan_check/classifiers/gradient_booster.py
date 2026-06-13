from pyspark.ml.classification import GBTClassifier

from .base_classifier import BaseClassifier


class GradientBoosterModel(BaseClassifier):
    @staticmethod
    def build_estimator():
        return GBTClassifier(
            featuresCol="features",
            labelCol="target",
            weightCol="weight",
            maxIter=50,
        )
