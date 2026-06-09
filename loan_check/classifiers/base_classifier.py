from abc import ABC, abstractmethod
from pyspark.ml import Pipeline, PipelineModel

class BaseClassifier(ABC):
    def __init__(self):
        self.estimator = self.build_estimator()
        self.model = None

    @abstractmethod
    def build_estimator(self):
        pass

    def train(self, df, feature_stages):
        pipeline = Pipeline(stages=[*feature_stages, self.estimator])
        self.model = pipeline.fit(df)
        return self.model
    

    def predict(self, df):
        return self.model.transform(df)

    def save_model(self, model_path):
        self.model.write().overwrite().save(str(model_path))

    def load_model(self, model_path):
        self.model = PipelineModel.load(str(model_path))
        return self.model        

        