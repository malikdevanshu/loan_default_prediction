from pyspark.ml.classification import DecisionTreeClassifier

from .base_classifier import BaseClassifier

class DecisionTreeClassifierModel(BaseClassifier):
    def build_estimator(self):
        return DecisionTreeClassifier(
            featuresCol="features",
            labelCol="target",
            maxDepth=8,
        )
    