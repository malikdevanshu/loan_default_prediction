from pathlib import Path
from pyspark.sql import SparkSession, DataFrame


class LendingClubPipeline:
    def __init__(self, app_name="lendingclub"):
        self.spark = self._get_spark(app_name)


    def _get_spark(self, app_name: str) -> SparkSession:
        return (
            SparkSession.builder
            .appName(app_name)
            .getOrCreate()
        )

 
    def load_data(self, data_path) -> DataFrame:
        data_path = Path(data_path)

        csv_file = next(data_path.rglob("*accepted*.csv"), None)

        if csv_file is None:
            raise FileNotFoundError(
                f"No accepted CSV found under: {data_path}"
            )

        print(f"Loading file: {csv_file.name}")

        df = (
            self.spark.read
            .option("header", True)
            .option("inferSchema", False)
            .option("multiLine", True)
            .option("quote", '"')
            .option("escape", '"')
            .csv(str(csv_file))
        )

        return df


    def save_bronze(self, df: DataFrame, output_name="accepted_loans"):
        bronze_path = Path(__file__).resolve().parents[1] / "data" / "bronze" / output_name

        bronze_path.mkdir(parents=True, exist_ok=True)

        (
            df.write
            .mode("overwrite")
            .parquet(str(bronze_path))
        )

        print(f"Saved bronze layer at: {bronze_path}")

        return bronze_path



