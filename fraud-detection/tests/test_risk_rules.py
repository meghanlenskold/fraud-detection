import pandas as pd
import pytest
from analyze_fraud import summarize_results, score_transactions
from features import build_model_frame
from risk_rules import label_risk, score_transaction


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

BASE_TX = {
    "device_risk_score": 10,
    "is_international": 0,
    "amount_usd": 100,
    "velocity_24h": 1,
    "failed_logins_24h": 0,
    "prior_chargebacks": 0,
}


# ---------------------------------------------------------------------------
# label_risk — boundary values
# ---------------------------------------------------------------------------

class TestLabelRisk:
    def test_zero_is_low(self):
        assert label_risk(0) == "low"

    def test_just_below_medium_threshold(self):
        assert label_risk(29) == "low"

    def test_medium_threshold_exact(self):
        assert label_risk(30) == "medium"

    def test_mid_medium(self):
        assert label_risk(45) == "medium"

    def test_just_below_high_threshold(self):
        assert label_risk(59) == "medium"

    def test_high_threshold_exact(self):
        assert label_risk(60) == "high"

    def test_max_score_is_high(self):
        assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# score_transaction — exact points per rule branch
# ---------------------------------------------------------------------------

class TestScoreTransactionDeviceRisk:
    def test_low_device_score_no_points(self):
        assert score_transaction({**BASE_TX, "device_risk_score": 39}) == 0

    def test_mid_device_score_adds_10(self):
        assert score_transaction({**BASE_TX, "device_risk_score": 40}) == 10
        assert score_transaction({**BASE_TX, "device_risk_score": 69}) == 10

    def test_high_device_score_adds_25(self):
        assert score_transaction({**BASE_TX, "device_risk_score": 70}) == 25
        assert score_transaction({**BASE_TX, "device_risk_score": 99}) == 25


class TestScoreTransactionInternational:
    def test_domestic_no_points(self):
        assert score_transaction({**BASE_TX, "is_international": 0}) == 0

    def test_international_adds_15(self):
        assert score_transaction({**BASE_TX, "is_international": 1}) == 15


class TestScoreTransactionAmount:
    def test_small_amount_no_points(self):
        assert score_transaction({**BASE_TX, "amount_usd": 499}) == 0

    def test_mid_amount_adds_10(self):
        assert score_transaction({**BASE_TX, "amount_usd": 500}) == 10
        assert score_transaction({**BASE_TX, "amount_usd": 999}) == 10

    def test_large_amount_adds_25(self):
        assert score_transaction({**BASE_TX, "amount_usd": 1000}) == 25
        assert score_transaction({**BASE_TX, "amount_usd": 5000}) == 25


class TestScoreTransactionVelocity:
    def test_low_velocity_no_points(self):
        assert score_transaction({**BASE_TX, "velocity_24h": 2}) == 0

    def test_mid_velocity_adds_5(self):
        assert score_transaction({**BASE_TX, "velocity_24h": 3}) == 5
        assert score_transaction({**BASE_TX, "velocity_24h": 5}) == 5

    def test_high_velocity_adds_20(self):
        assert score_transaction({**BASE_TX, "velocity_24h": 6}) == 20
        assert score_transaction({**BASE_TX, "velocity_24h": 15}) == 20


class TestScoreTransactionFailedLogins:
    def test_no_failed_logins_no_points(self):
        assert score_transaction({**BASE_TX, "failed_logins_24h": 0}) == 0
        assert score_transaction({**BASE_TX, "failed_logins_24h": 1}) == 0

    def test_mid_failed_logins_adds_10(self):
        assert score_transaction({**BASE_TX, "failed_logins_24h": 2}) == 10
        assert score_transaction({**BASE_TX, "failed_logins_24h": 4}) == 10

    def test_high_failed_logins_adds_20(self):
        assert score_transaction({**BASE_TX, "failed_logins_24h": 5}) == 20
        assert score_transaction({**BASE_TX, "failed_logins_24h": 10}) == 20


class TestScoreTransactionPriorChargebacks:
    def test_no_chargebacks_no_points(self):
        assert score_transaction({**BASE_TX, "prior_chargebacks": 0}) == 0

    def test_one_chargeback_adds_5(self):
        assert score_transaction({**BASE_TX, "prior_chargebacks": 1}) == 5

    def test_two_or_more_chargebacks_adds_20(self):
        assert score_transaction({**BASE_TX, "prior_chargebacks": 2}) == 20
        assert score_transaction({**BASE_TX, "prior_chargebacks": 5}) == 20


class TestScoreTransactionClamping:
    def test_clean_transaction_scores_zero(self):
        assert score_transaction(BASE_TX) == 0

    def test_score_cannot_exceed_100(self):
        tx = {
            "device_risk_score": 85,
            "is_international": 1,
            "amount_usd": 1500,
            "velocity_24h": 10,
            "failed_logins_24h": 8,
            "prior_chargebacks": 3,
        }
        # Raw sum = 25+15+25+20+20+20 = 125, must clamp to 100
        assert score_transaction(tx) == 100

    def test_worst_case_is_high_risk(self):
        tx = {
            "device_risk_score": 85,
            "is_international": 1,
            "amount_usd": 1400,
            "velocity_24h": 8,
            "failed_logins_24h": 7,
            "prior_chargebacks": 2,
        }
        assert label_risk(score_transaction(tx)) == "high"


# ---------------------------------------------------------------------------
# build_model_frame — feature engineering
# ---------------------------------------------------------------------------

class TestBuildModelFrame:
    @pytest.fixture
    def minimal_data(self):
        transactions = pd.DataFrame([
            {"transaction_id": 1, "account_id": 10, "amount_usd": 1200, "failed_logins_24h": 0},
            {"transaction_id": 2, "account_id": 10, "amount_usd": 800,  "failed_logins_24h": 1},
            {"transaction_id": 3, "account_id": 10, "amount_usd": 400,  "failed_logins_24h": 2},
            {"transaction_id": 4, "account_id": 10, "amount_usd": 50,   "failed_logins_24h": 5},
        ])
        accounts = pd.DataFrame([{"account_id": 10, "prior_chargebacks": 0}])
        return transactions, accounts

    def test_is_large_amount_flag(self, minimal_data):
        transactions, accounts = minimal_data
        df = build_model_frame(transactions, accounts)
        assert df.loc[df["transaction_id"] == 1, "is_large_amount"].iloc[0] == 1
        assert df.loc[df["transaction_id"] == 2, "is_large_amount"].iloc[0] == 0

    def test_is_large_amount_boundary(self, minimal_data):
        transactions, accounts = minimal_data
        df = build_model_frame(transactions, accounts)
        # $800 is below the $1000 threshold
        assert df.loc[df["transaction_id"] == 2, "is_large_amount"].iloc[0] == 0
        # $1200 meets the threshold
        assert df.loc[df["transaction_id"] == 1, "is_large_amount"].iloc[0] == 1

    def test_login_pressure_none(self, minimal_data):
        transactions, accounts = minimal_data
        df = build_model_frame(transactions, accounts)
        assert df.loc[df["transaction_id"] == 1, "login_pressure"].iloc[0] == "none"

    def test_login_pressure_low(self, minimal_data):
        transactions, accounts = minimal_data
        df = build_model_frame(transactions, accounts)
        assert df.loc[df["transaction_id"] == 3, "login_pressure"].iloc[0] == "low"

    def test_login_pressure_high(self, minimal_data):
        transactions, accounts = minimal_data
        df = build_model_frame(transactions, accounts)
        assert df.loc[df["transaction_id"] == 4, "login_pressure"].iloc[0] == "high"

    def test_account_columns_merged(self, minimal_data):
        transactions, accounts = minimal_data
        df = build_model_frame(transactions, accounts)
        assert "prior_chargebacks" in df.columns
        assert (df["prior_chargebacks"] == 0).all()


# ---------------------------------------------------------------------------
# summarize_results — pipeline metrics
# ---------------------------------------------------------------------------

class TestSummarizeResults:
    @pytest.fixture
    def scored_df(self):
        return pd.DataFrame([
            {"transaction_id": 1, "risk_label": "high",   "amount_usd": 500},
            {"transaction_id": 2, "risk_label": "high",   "amount_usd": 300},
            {"transaction_id": 3, "risk_label": "medium", "amount_usd": 200},
            {"transaction_id": 4, "risk_label": "low",    "amount_usd": 100},
            {"transaction_id": 5, "risk_label": "low",    "amount_usd": 50},
        ])

    @pytest.fixture
    def chargebacks_df(self):
        # tx 1 and 3 are confirmed chargebacks
        return pd.DataFrame([
            {"transaction_id": 1},
            {"transaction_id": 3},
        ])

    def test_chargeback_rate_high_bucket(self, scored_df, chargebacks_df):
        summary = summarize_results(scored_df, chargebacks_df)
        high_row = summary.loc[summary["risk_label"] == "high"].iloc[0]
        # 1 chargeback out of 2 high-risk transactions
        assert high_row["chargeback_rate"] == pytest.approx(0.5)

    def test_chargeback_rate_medium_bucket(self, scored_df, chargebacks_df):
        summary = summarize_results(scored_df, chargebacks_df)
        medium_row = summary.loc[summary["risk_label"] == "medium"].iloc[0]
        assert medium_row["chargeback_rate"] == pytest.approx(1.0)

    def test_chargeback_rate_low_bucket(self, scored_df, chargebacks_df):
        summary = summarize_results(scored_df, chargebacks_df)
        low_row = summary.loc[summary["risk_label"] == "low"].iloc[0]
        assert low_row["chargeback_rate"] == pytest.approx(0.0)

    def test_transaction_counts(self, scored_df, chargebacks_df):
        summary = summarize_results(scored_df, chargebacks_df)
        counts = summary.set_index("risk_label")["transactions"].to_dict()
        assert counts == {"high": 2, "medium": 1, "low": 2}

    def test_total_amount_usd(self, scored_df, chargebacks_df):
        summary = summarize_results(scored_df, chargebacks_df)
        high_row = summary.loc[summary["risk_label"] == "high"].iloc[0]
        assert high_row["total_amount_usd"] == pytest.approx(800.0)

    def test_avg_amount_usd(self, scored_df, chargebacks_df):
        summary = summarize_results(scored_df, chargebacks_df)
        low_row = summary.loc[summary["risk_label"] == "low"].iloc[0]
        assert low_row["avg_amount_usd"] == pytest.approx(75.0)

    def test_no_chargebacks_produces_zero_rate(self, scored_df):
        no_cb = pd.DataFrame(columns=["transaction_id"])
        summary = summarize_results(scored_df, no_cb)
        assert (summary["chargeback_rate"] == 0).all()
