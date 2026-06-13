from pyspark.ml.classification import DecisionTreeClassifier

from .base_classifier import BaseClassifier


class DecisionTreeClassifierModel(BaseClassifier):
    @staticmethod
    def build_estimator():
        return DecisionTreeClassifier(
            featuresCol="features",
            labelCol="target",
            weightCol="weight",
            maxDepth=8,
        )
