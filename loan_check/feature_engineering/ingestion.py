from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from loan_check.feature_engineering.preprocessing import Preprocessor


class LendingClubPipeline:
    def __init__(self, app_name: str = "lendingclub") -> None:
        self.spark = self._get_spark(app_name)

    @staticmethod
    def _get_spark(app_name: str) -> SparkSession:
        return (
            SparkSession.builder.appName(app_name)
            .master("local[8]")
            .config("spark.driver.memory", "8g")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.sql.adaptive.enabled", "true")
            .getOrCreate()
        )

    @staticmethod
    def _is_written(path: Path) -> bool:
        return (path / "_SUCCESS").exists()

    def load_raw_csv(self, raw_data_path: Path) -> DataFrame:
        raw_data_path = Path(raw_data_path)

        csv_file = next(
            (f for f in raw_data_path.rglob("*accepted*.csv") if f.is_file()),
            None,
        )

        if csv_file is None:
            msg = f"No accepted CSV found under: {raw_data_path}"
            raise FileNotFoundError(msg)

        print(f"Loading raw CSV: {csv_file.name}")

        return (
            self.spark.read.option("header", True)
            .option("inferSchema", False)
            .option("multiLine", True)
            .option("quote", '"')
            .option("escape", '"')
            .csv(str(csv_file))
        )

    def build_bronze(
        self, raw_data_path: Path, bronze_path: Path, overwrite: bool = False
    ) -> DataFrame:
        bronze_path = Path(bronze_path)

        if self._is_written(bronze_path) and not overwrite:
            print(f"Bronze exists, loading parquet: {bronze_path}")
            return self.spark.read.parquet(str(bronze_path))

        df = self.load_raw_csv(raw_data_path)

        bronze_path.parent.mkdir(parents=True, exist_ok=True)
        df.write.mode("overwrite").parquet(str(bronze_path))
        print(f"Saved bronze layer at: {bronze_path}")

        return self.spark.read.parquet(str(bronze_path))

    def build_silver(
        self,
        bronze_df: DataFrame,
        silver_path: Path,
        preprocessor: Preprocessor,
        overwrite: bool = False,
    ) -> DataFrame:
        silver_path = Path(silver_path)

        if self._is_written(silver_path) and not overwrite:
            print(f"Silver exists, loading parquet: {silver_path}")
            return self.spark.read.parquet(str(silver_path))

        df = preprocessor.preprocess(bronze_df)

        silver_path.parent.mkdir(parents=True, exist_ok=True)
        df.write.mode("overwrite").parquet(str(silver_path))
        print(f"Saved silver layer at: {silver_path}")

        return self.spark.read.parquet(str(silver_path))
