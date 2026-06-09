COLS_TO_DROP = [
    "id", "member_id", "url", "zip_code", "policy_code", "pymnt_plan",
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "hardship_flag", "hardship_type",
    "hardship_reason", "hardship_status", "deferral_term", "hardship_amount",
    "hardship_start_date", "hardship_end_date", "payment_plan_start_date",
    "hardship_length", "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage",
    "settlement_term", "annual_inc_joint", "dti_joint",
    "verification_status_joint", "revol_bal_joint", "sec_app_fico_range_low",
    "sec_app_fico_range_high", "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
    "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog", "mths_since_last_record",
    "mths_since_last_major_derog", "desc", "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq",
]


ENCODED_RAW_COLS = [
    "loan_status",       # -> target
    "term",              # -> term_months
    "grade",             # -> grade_ord (then dropped; sub_grade_ord supersedes)
    "sub_grade",         # -> sub_grade_ord
    "emp_length",        # -> emp_length_num
    "emp_title",         # -> has_emp_title
    "issue_d",           # -> issue_year / credit_hist_months
    "earliest_cr_line",  # -> credit_hist_months
    "funded_amnt",       # collinear with loan_amnt
    "funded_amnt_inv",
]

DROP_STRUCTURAL = [
    "il_util", "mths_since_rcnt_il", "all_util", "open_acc_6m",
    "total_cu_tl", "inq_last_12m", "open_act_il", "open_il_12m",
    "open_il_24m", "total_bal_il", "open_rv_12m", "open_rv_24m",
    "max_bal_bc", "inq_fi",
]


RECENCY_COLS = [
    "mths_since_last_delinq", "mths_since_recent_inq",
    "mths_since_recent_bc", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
]


ZERO_FILL_COLS = [
    "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_accts_ever_120_pd", "tot_coll_amt",
]


FLAG_BUT_IMPUTE_COLS = [
    "revol_util", "bc_util", "tot_cur_bal", "avg_cur_bal",
]


NOMINAL_COLS = [
    "home_ownership", "verification_status", "purpose", "addr_state",
    "initial_list_status", "application_type", "disbursement_method",
]


NUMERIC_COLS = [
    "loan_amnt", "installment", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal",
    "total_acc", "collections_12_mths_ex_med", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
    "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
    "mort_acc", "mths_since_recent_bc", "mths_since_recent_inq",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
    "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
    "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
]

PAID_STATUSES = [
    "Fully Paid", "Does not meet the credit policy. Status:Fully Paid",
]
DEFAULT_STATUSES = [
    "Charged Off", "Default",
    "Does not meet the credit policy. Status:Charged Off",
]