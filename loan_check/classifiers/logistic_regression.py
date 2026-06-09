from pyspark.ml.classification import  LogisticRegression

from .base_classifier import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):
    def build_estimator(self):
        return LogisticRegression(
            featuresCol= "features",
            labelCol= "target",
            weightCol="weight",
            maxIter = 100,
        )