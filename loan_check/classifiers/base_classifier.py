from abc import ABC, abstractmethod
from typing import Any

from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import DataFrame


class BaseClassifier(ABC):
    def __init__(self) -> None:
        self.estimator = self.build_estimator()
        self.model: PipelineModel | None = None

    @abstractmethod
    def build_estimator(self) -> Any:
        pass

    def train(self, df: DataFrame, feature_stages: list[Any]) -> PipelineModel:
        pipeline = Pipeline(stages=[*feature_stages, self.estimator])
        self.model = pipeline.fit(df)
        return self.model

    def predict(self, df: DataFrame) -> DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        return self.model.transform(df)

    def save_model(self, model_path: str) -> None:
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        self.model.write().overwrite().save(str(model_path))

    def load_model(self, model_path: str) -> PipelineModel:
        self.model = PipelineModel.load(str(model_path))
        return self.model
