from pyspark.sql import functions as F
from feature_config import load_feature_config


class Preprocessor:
    def __init__(self):
        self.config = load_feature_config()
        self.paid = self.config["paid_statuses"]
        self.default = self.config["default_statuses"]
        self.emp_length_map = self.config["emp_length_map"]

    def _add_target(self, df):
        df = df.filter(F.col("loan_status").isin(self.paid + self.default))
        return df.withColumn(
            "target", F.col("loan_status").isin(self.default).cast("int")
        )

    def _parse_string_numbers(self, df):
        for c in ("int_rate", "revol_util"):
            df = df.withColumn(
                c, F.regexp_replace(F.col(c), "%", "").cast("double")
            )
        return df.withColumn(
            "term_months", F.regexp_extract("term", r"(\d+)", 1).cast("int")
        )

    def _add_date_features(self, df):
        return (
            df
            .withColumn("issue_d", F.to_date("issue_d", "MMM-yyyy"))
            .withColumn(
                "earliest_cr_line", F.to_date("earliest_cr_line", "MMM-yyyy")
            )
            .withColumn(
                "credit_hist_months",
                F.months_between("issue_d", "earliest_cr_line"),
            )
            .withColumn("issue_year", F.year("issue_d"))
        )

    def _add_ordinal_features(self, df):
        emp_expr = F.create_map(
            [F.lit(x) for kv in self.emp_length_map.items() for x in kv]
        )
        return (
            df
            .withColumn("emp_length_num", emp_expr[F.col("emp_length")])
            # grade_ord is created then dropped (sub_grade_ord supersedes it)
            .withColumn("grade_ord", F.expr("ascii(grade) - ascii('A') + 1"))
            .withColumn(
                "sub_grade_ord",
                (
                    F.expr("ascii(substring(sub_grade,1,1)) - ascii('A')")
                    * F.lit(5)
                )
                + F.substring("sub_grade", 2, 1).cast("int"),
            )
        )

    def _add_text_flags(self, df):
        df = df.withColumn(
            "has_emp_title",
            (
                F.length(F.trim(F.coalesce(F.col("emp_title"), F.lit(""))))
                > 0
            ).cast("int"),
        )
        return df.withColumn("title", F.coalesce(F.lower("title"), F.lit("")))

    def _cast_numeric(self, df):
        for c in self.config["numeric_columns"]:
            if c in df.columns:
                df = df.withColumn(c, F.col(c).cast("double"))
        return df

    def _add_ratio_features(self, df):
        def safe_div(num, den):
            return F.when(F.col(den) > 0, F.col(num) / F.col(den))

        return (
            df
            .withColumn(
                "fico_avg",
                (F.col("fico_range_low") + F.col("fico_range_high")) / 2,
            )
            .withColumn("loan_to_income", safe_div("loan_amnt", "annual_inc"))
            .withColumn(
                "installment_to_income",
                F.when(
                    F.col("annual_inc") > 0,
                    (F.col("installment") * 12) / F.col("annual_inc"),
                ),
            )
            .withColumn(
                "open_to_total_acc", safe_div("open_acc", "total_acc")
            )
            .withColumn(
                "has_delinq_2yrs", (F.col("delinq_2yrs") > 0).cast("int")
            )
            .withColumn("has_pub_rec", (F.col("pub_rec") > 0).cast("int"))
        )

    def _add_structural_flag(self, df):
        df = df.withColumn(
            "has_secondary_data",
            F.col("open_il_24m").isNotNull().cast("int"),
        )
        return df.drop(*self.config["structural_drop_columns"])

    def _handle_missing(self, df):
        for c in self.config["recency_columns"] + self.config["flag_and_impute_columns"]:
            if c in df.columns:
                df = df.withColumn(
                    f"{c}_missing", F.col(c).isNull().cast("int")
                )

        zero_cols = [
            c for c in self.config["recency_columns"] + self.config["zero_fill_columns"] if c in df.columns
        ]
        return df.fillna(0, subset=zero_cols)

    def preprocess(self, data):
        if "loan_status" not in data.columns:
            raise ValueError("data must contain the loan_status column")

        df = data.drop(*self.config["drop_columns"])
        df = self._add_target(df)
        df = self._parse_string_numbers(df)
        df = self._add_date_features(df)
        df = self._add_ordinal_features(df)
        df = self._add_text_flags(df)
        df = df.drop(*self.config["encoded_raw_columns"])
        df = self._cast_numeric(df)
        df = self._add_ratio_features(df)
        df = df.drop("fico_range_low", "fico_range_high", "grade_ord")
        df = self._add_structural_flag(df)
        df = self._handle_missing(df)
        return df