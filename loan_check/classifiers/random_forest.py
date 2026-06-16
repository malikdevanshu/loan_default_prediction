from pyspark.ml.classification import RandomForestClassifier

from .base_classifier import BaseClassifier


class RandomForestClassifierModel(BaseClassifier):
    @staticmethod
    def build_estimator() -> RandomForestClassifier:
        return RandomForestClassifier(
            featuresCol="features",
            labelCol="target",
            weightCol="weight",
            numTrees=100,
        )
