import pytest

from loan_check.feature_engineering.preprocessing import Preprocessor


def test_preprocess_requires_loan_status(spark):
    df = spark.createDataFrame([("x",)], "other string")
    with pytest.raises(ValueError):
        Preprocessor().preprocess(df)


def test_current_status_is_filtered(preprocessed_df, rows_by_amnt):
    assert preprocessed_df.count() == 4
    assert 8000 not in rows_by_amnt


def test_target_labels(rows_by_amnt):
    assert rows_by_amnt[10000]["target"] == 0   # Fully Paid
    assert rows_by_amnt[20000]["target"] == 1   # Charged Off
    assert rows_by_amnt[5000]["target"] == 0    # Does not meet ... Fully Paid
    assert rows_by_amnt[30000]["target"] == 1   # Default


def test_fico_avg(rows_by_amnt):
    assert rows_by_amnt[10000]["fico_avg"] == pytest.approx(690.0)


def test_term_months(rows_by_amnt):
    assert rows_by_amnt[10000]["term_months"] == 36
    assert rows_by_amnt[20000]["term_months"] == 60


def test_emp_length_ordinal(rows_by_amnt):
    assert rows_by_amnt[10000]["emp_length_num"] == 10   # "10+ years"
    assert rows_by_amnt[20000]["emp_length_num"] == 0    # "< 1 year"


def test_sub_grade_ordinal(rows_by_amnt):
    # (ascii(letter) - ascii('A')) * 5 + digit
    assert rows_by_amnt[10000]["sub_grade_ord"] == 8     # B3 -> 1*5 + 3
    assert rows_by_amnt[20000]["sub_grade_ord"] == 1     # A1 -> 0*5 + 1
    assert rows_by_amnt[30000]["sub_grade_ord"] == 25    # E5 -> 4*5 + 5


def test_ratio_features_and_safe_div(rows_by_amnt):
    assert rows_by_amnt[10000]["loan_to_income"] == pytest.approx(0.2)
    # B has annual_inc == 0, so the guarded division returns null.
    assert rows_by_amnt[20000]["loan_to_income"] is None
    assert rows_by_amnt[20000]["installment_to_income"] is None


def test_binary_flags(rows_by_amnt):
    assert rows_by_amnt[10000]["has_pub_rec"] == 1        # pub_rec 1
    assert rows_by_amnt[20000]["has_pub_rec"] == 0        # pub_rec 0
    assert rows_by_amnt[10000]["has_delinq_2yrs"] == 0    # delinq 0
    assert rows_by_amnt[20000]["has_delinq_2yrs"] == 1    # delinq 3
    assert rows_by_amnt[10000]["has_secondary_data"] == 1  # open_il_24m "2"
    assert rows_by_amnt[20000]["has_secondary_data"] == 0  # open_il_24m null


def test_text_flags(rows_by_amnt):
    assert rows_by_amnt[10000]["has_emp_title"] == 1          # "engineer"
    assert rows_by_amnt[20000]["has_emp_title"] == 0          # null
    assert rows_by_amnt[10000]["title"] == "debt consolidation"  # lowered
    assert rows_by_amnt[20000]["title"] == ""                # null -> ""


def test_missing_indicators_and_zero_fill(rows_by_amnt):
    assert rows_by_amnt[10000]["revol_util_missing"] == 0
    assert rows_by_amnt[20000]["revol_util_missing"] == 1
    assert rows_by_amnt[20000]["mths_since_last_delinq_missing"] == 1
    # recency column is also zero-filled after the flag is recorded
    assert rows_by_amnt[20000]["mths_since_last_delinq"] == pytest.approx(0.0)


def test_leakage_and_raw_columns_dropped(preprocessed_df):
    cols = set(preprocessed_df.columns)
    must_be_absent = {
        # post-origination leakage
        "recoveries", "total_pymnt",
        # identifiers / raw encoded columns
        "id", "loan_status", "grade", "sub_grade", "emp_length", "emp_title",
        "term", "issue_d", "earliest_cr_line",
        # structural column consumed by has_secondary_data then dropped
        "open_il_24m",
    }
    assert must_be_absent.isdisjoint(cols), must_be_absent & cols


def test_expected_columns_present(preprocessed_df):
    cols = set(preprocessed_df.columns)
    expected = {
        "target", "loan_amnt", "fico_avg", "term_months", "emp_length_num",
        "sub_grade_ord", "loan_to_income", "credit_hist_months", "issue_year",
        "has_secondary_data", "title",
        # nominal columns survive for the feature pipeline
        "home_ownership", "purpose", "addr_state",
    }
    assert expected.issubset(cols), expected - cols


def test_malformed_date_becomes_null(spark):
    """A malformed date must not crash preprocessing.

    Spark 4 runs in ANSI mode, where F.to_date throws on unparseable input;
    _add_date_features uses F.try_to_date, which yields null instead so the
    row survives.
    """
    from tests.conftest import RAW_COLUMNS, RAW_ROWS

    bad = list(RAW_ROWS[0])
    bad[RAW_COLUMNS.index("issue_d")] = "not-a-date"
    df = spark.createDataFrame(
        [tuple(bad)],
        ",".join(f"{c} string" for c in RAW_COLUMNS),
    )
    out = Preprocessor().preprocess(df).collect()
    assert len(out) == 1
    assert out[0]["issue_year"] is None
    assert out[0]["credit_hist_months"] is None