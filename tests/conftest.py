import os
import sys

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ["PATH"] = os.path.join(os.environ["HADOOP_HOME"], "bin") + os.pathsep + os.environ["PATH"]
import pytest

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("loan-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


RAW_COLUMNS = [
    "id", "loan_status", "loan_amnt", "int_rate", "revol_util", "term",
    "issue_d", "earliest_cr_line", "emp_length", "grade", "sub_grade",
    "emp_title", "title", "fico_range_low", "fico_range_high", "annual_inc",
    "installment", "open_acc", "total_acc", "delinq_2yrs", "pub_rec",
    "open_il_24m", "mths_since_last_delinq", "num_tl_30dpd", "recoveries",
    "total_pymnt", "home_ownership", "verification_status", "purpose",
    "addr_state", "initial_list_status", "application_type",
    "disbursement_method",
]

RAW_ROWS = [
    ("A1", "Fully Paid", "10000", "12.5%", "30%", " 36 months", "Dec-2015",
     "Dec-2005", "10+ years", "B", "B3", "engineer", "Debt Consolidation",
     "680", "700", "50000", "300", "5", "10", "0", "1", "2", "12", "0",
     "0", "10500", "RENT", "Verified", "debt_consolidation", "CA", "w",
     "Individual", "Cash"),
    ("B1", "Charged Off", "20000", "18.0%", None, " 60 months", "Mar-2014",
     "Jun-2000", "< 1 year", "A", "A1", None, None, "660", "680", "0", "520",
     "8", "20", "3", "0", None, None, "1", "500", "3000", "MORTGAGE",
     "Not Verified", "credit_card", "NY", "f", "Individual", "Cash"),
    ("C1", "Current", "8000", "9.9%", "12%", " 36 months", "Jan-2018",
     "Feb-2010", "5 years", "C", "C2", "manager", "home improvement", "720",
     "740", "90000", "250", "6", "15", "0", "0", "0", "36", "0", "0", "9000",
     "OWN", "Source Verified", "home_improvement", "TX", "w", "Individual",
     "Cash"),
    ("D1", "Does not meet the credit policy. Status:Fully Paid", "5000",
     "11.1%", "30%", " 36 months", "May-2011", "Sep-2003", "3 years", "C",
     "C1", "nurse", "vacation", "700", "720", "55000", "160", "9", "18", "0",
     "0", "0", "12", "0", "0", "5200", "RENT", "Verified", "vacation", "FL",
     "w", "Individual", "Cash"),
    ("E1", "Default", "30000", "22.4%", "85%", " 60 months", "Jul-2013",
     "Aug-1999", "2 years", "E", "E5", "driver", "other", "610", "620",
     "40000", "800", "20", "40", "2", "0", "1", "6", "1", "800", "4000",
     "MORTGAGE", "Verified", "other", "WA", "f", "Joint App", "DirectPay"),
]


@pytest.fixture(scope="session")
def raw_df(spark):
    schema = StructType([StructField(c, StringType()) for c in RAW_COLUMNS])
    return spark.createDataFrame(RAW_ROWS, schema=schema)


@pytest.fixture(scope="session")
def preprocessed_df(raw_df):
    from loan_check.feature_engineering.preprocessing import Preprocessor

    return Preprocessor().preprocess(raw_df).cache()


@pytest.fixture(scope="session")
def rows_by_amnt(preprocessed_df):
    return {int(r["loan_amnt"]): r for r in preprocessed_df.collect()}