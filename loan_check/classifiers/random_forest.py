from pyspark.ml.classification import RandomForestClassifier

from .base_classifier import BaseClassifier

class RandomForestClassifierModel(BaseClassifier):
    def build_estimator(self):
        return RandomForestClassifier(
            featuresCol="features", 
            labelCol="target",
            numTrees=100
        )