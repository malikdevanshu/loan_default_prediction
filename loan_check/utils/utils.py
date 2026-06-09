from pathlib import Path
 
from pyspark.ml.feature import (
    IDF,
    CountVectorizer,
    Imputer,
    OneHotEncoder,
    RegexTokenizer,
    StopWordsRemover,
    StringIndexer,
    VectorAssembler,
)

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
 
from loan_check.classifiers.decision_tree import DecisionTreeClassifierModel
from loan_check.classifiers.gradient_booster import GradientBoosterModel
from loan_check.classifiers.logistic_regression import (
    LogisticRegressionClassifier,
)
from loan_check.classifiers.random_forest import RandomForestClassifierModel
from config.config import load_config
from loan_check.feature_engineering.feature_config import load_feature_config
from loan_check.feature_engineering.ingestion import LendingClubPipeline
from loan_check.feature_engineering.preprocessing import Preprocessor

def get_config_values():
    config = load_config()
 
    return {
        "raw_data_path": Path(config["paths"]["raw_data"]),
        "bronze_data_path": Path(config["paths"]["bronze_data"]),
        "silver_data_path": Path(config["paths"]["silver_data"]),
        "model_dir": Path(config["model_path"]["path"]),
        "test_size": config["test_size"]["size"],
        "random_state": config["random_state"]["state"],
    }

def get_models():
    return {
        "logistic_regression": LogisticRegressionClassifier,
        "decision_tree": DecisionTreeClassifierModel,
        "random_forest": RandomForestClassifierModel,
        "gradient_boosted": GradientBoosterModel,
    }
def build_param_grid(model_name, estimator):
    if model_name == "logistic_regression":
        grid = (
            ParamGridBuilder()
            .addGrid(estimator.regParam, [0.0, 0.01, 0.1])
            .addGrid(estimator.elasticNetParam, [0.0, 0.5])
        )
    elif model_name == "decision_tree":
        grid = (
            ParamGridBuilder().addGrid(estimator.maxDepth, [5, 8, 12])
        )
    elif model_name == "random_forest":
        grid = (
            ParamGridBuilder()
            .addGrid(estimator.numTrees, [50, 100])
            .addGrid(estimator.maxDepth, [5, 10])
        )
    elif model_name == "gradient_boosted":
        grid = (
            ParamGridBuilder()
            .addGrid(estimator.maxIter, [20, 50])
            .addGrid(estimator.maxDepth, [3, 5])
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return grid.build()

def build_feature_stages(df):
    nominal_cols = load_feature_config()["features"]["nominal_columns"]
 
    indexers = [
        StringIndexer(
            inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"
        )
        for c in nominal_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in nominal_cols
    ]
 
    title_tok = RegexTokenizer(
        inputCol="title", outputCol="title_tok", pattern=r"\W+"
    )
    title_stop = StopWordsRemover(
        inputCol="title_tok", outputCol="title_tok_clean"
    )
    title_cv = CountVectorizer(
        inputCol="title_tok_clean",
        outputCol="title_tf",
        vocabSize=200,
        minDF=20,
    )
    title_idf = IDF(inputCol="title_tf", outputCol="title_tfidf")
 
    numeric_inputs = [
        f.name
        for f in df.schema.fields
        if isinstance(f.dataType, NumericType)
        and f.name not in ("target", "weight")
    ]
 
    # medians are learned when the pipeline is fit on the train split, so
    # there is no leakage from the test rows.
    imputer = Imputer(
        inputCols=numeric_inputs,
        outputCols=numeric_inputs,
        strategy="median",
    )
 
    assembler = VectorAssembler(
        inputCols=[
            *numeric_inputs,
            *[f"{c}_ohe" for c in nominal_cols],
            "title_tfidf",
        ],
        outputCol="features",
        handleInvalid="keep",
    )
 
    return [
        imputer,
        *indexers,
        *encoders,
        title_tok,
        title_stop,
        title_cv,
        title_idf,
        assembler,
    ]

def add_class_weights(df, label_col="target", weight_col="weight"):
    counts = df.groupBy(label_col).count().collect()
    total = sum(row["count"] for row in counts)
    n_classes = len(counts)

    weights = {
        row[label_col]: total / (n_classes * row["count"])
        for row in counts
    }

    weight_expr = F.create_map(
        [F.lit(x) for pair in weights.items() for x in pair]
    )
    return df.withColumn(weight_col, weight_expr[F.col(label_col)])

def load_and_prepare_data():
    config_values = get_config_values()
 
    pipeline = LendingClubPipeline()
 
    bronze = pipeline.build_bronze(
        raw_data_path=config_values["raw_data_path"],
        bronze_path=config_values["bronze_data_path"],
    )
 
    preprocessor = Preprocessor()
    silver = pipeline.build_silver(
        bronze_df=bronze,
        silver_path=config_values["silver_data_path"],
        preprocessor=preprocessor,
    )
 
    return silver.randomSplit(
        [1 - config_values["test_size"], config_values["test_size"]],
        seed=config_values["random_state"],
    )
 
 
def get_model_paths(model_name, model_type):
    config_values = get_config_values()
    model_dir = config_values["model_dir"]
 
    return model_dir / f"{model_name}_{model_type}"