from pyspark.ml.classification import LogisticRegression

from .base_classifier import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    @staticmethod
    def build_estimator():
        return LogisticRegression(
            featuresCol="features",
            labelCol="target",
            weightCol="weight",
            maxIter=100,
        )
