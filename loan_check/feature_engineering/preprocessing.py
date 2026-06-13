from pyspark.sql import functions as f

from loan_check.feature_engineering.feature_config import load_feature_config


class Preprocessor:
    def __init__(self):
        self.config = load_feature_config()
        self.paid = self.config["features"]["target_labels"]["paid_statuses"]
        self.default = self.config["features"]["target_labels"][
            "default_statuses"
        ]
        self.emp_length_map = self.config["features"]["emp_length_map"]

    def _add_target(self, df):
        df = df.filter(f.col("loan_status").isin(self.paid + self.default))
        return df.withColumn(
            "target", f.col("loan_status").isin(self.default).cast("int")
        )

    @staticmethod
    def _parse_string_numbers(df):
        for c in ("int_rate", "revol_util"):
            df = df.withColumn(
                c, f.regexp_replace(f.col(c), "%", "").cast("double")
            )
        return df.withColumn(
            "term_months", f.regexp_extract("term", r"(\d+)", 1).cast("int")
        )

    @staticmethod
    def _add_date_features(df):
        return (
            df.withColumn("issue_d", f.try_to_date("issue_d", "MMM-yyyy"))
            .withColumn(
                "earliest_cr_line",
                f.try_to_date("earliest_cr_line", "MMM-yyyy"),
            )
            .withColumn(
                "credit_hist_months",
                f.months_between("issue_d", "earliest_cr_line"),
            )
            .withColumn("issue_year", f.year("issue_d"))
        )

    def _add_ordinal_features(self, df):
        emp_expr = f.create_map(
            [f.lit(x) for kv in self.emp_length_map.items() for x in kv]
        )
        return (
            df.withColumn("emp_length_num", emp_expr[f.col("emp_length")])
            # grade_ord is created then dropped (sub_grade_ord supersedes it)
            .withColumn("grade_ord", f.expr("ascii(grade) - ascii('A') + 1"))
            .withColumn(
                "sub_grade_ord",
                (
                    f.expr("ascii(substring(sub_grade,1,1)) - ascii('A')")
                    * f.lit(5)
                )
                + f.substring("sub_grade", 2, 1).cast("int"),
            )
        )

    @staticmethod
    def _add_text_flags(df):
        df = df.withColumn(
            "has_emp_title",
            (
                f.length(f.trim(f.coalesce(f.col("emp_title"), f.lit("")))) > 0
            ).cast("int"),
        )
        return df.withColumn("title", f.coalesce(f.lower("title"), f.lit("")))

    def _cast_numeric(self, df):
        for c in self.config["features"]["numeric_columns"]:
            if c in df.columns:
                df = df.withColumn(c, f.col(c).cast("double"))
        return df

    @staticmethod
    def _add_ratio_features(df):
        def safe_div(num, den):
            return f.when(f.col(den) > 0, f.col(num) / f.col(den))

        return (
            df.withColumn(
                "fico_avg",
                (f.col("fico_range_low") + f.col("fico_range_high")) / 2,
            )
            .withColumn("loan_to_income", safe_div("loan_amnt", "annual_inc"))
            .withColumn(
                "installment_to_income",
                f.when(
                    f.col("annual_inc") > 0,
                    (f.col("installment") * 12) / f.col("annual_inc"),
                ),
            )
            .withColumn("open_to_total_acc", safe_div("open_acc", "total_acc"))
            .withColumn(
                "has_delinq_2yrs", (f.col("delinq_2yrs") > 0).cast("int")
            )
            .withColumn("has_pub_rec", (f.col("pub_rec") > 0).cast("int"))
        )

    def _add_structural_flag(self, df):
        df = df.withColumn(
            "has_secondary_data",
            f.col("open_il_24m").isNotNull().cast("int"),
        )
        return df.drop(*self.config["features"]["structural_drop_columns"])

    def _handle_missing(self, df):
        for c in (
            self.config["features"]["recency_columns"]
            + self.config["features"]["flag_and_impute_columns"]
        ):
            if c in df.columns:
                df = df.withColumn(
                    f"{c}_missing", f.col(c).isNull().cast("int")
                )

        zero_cols = [
            c
            for c in self.config["features"]["recency_columns"]
            + self.config["features"]["zero_fill_columns"]
            if c in df.columns
        ]
        return df.fillna(0, subset=zero_cols)

    def preprocess(self, data):
        if "loan_status" not in data.columns:
            raise ValueError("data must contain the loan_status column")

        df = data.drop(*self.config["features"]["drop_columns"])
        df = self._add_target(df)
        df = self._parse_string_numbers(df)
        df = self._add_date_features(df)
        df = self._add_ordinal_features(df)
        df = self._add_text_flags(df)
        df = df.drop(*self.config["features"]["encoded_raw_columns"])
        df = self._cast_numeric(df)
        df = self._add_ratio_features(df)
        df = df.drop("fico_range_low", "fico_range_high", "grade_ord")
        df = self._add_structural_flag(df)
        df = self._handle_missing(df)
        return df  # noqa: RET504
