from pyspark.ml.classification import  LogisticRegression

from .base_classifier import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):
    def build_estimator(self):
        return LogisticRegression(
            featureCol= "features",
            labelCol= "target",
            maxIter = 100,
        )