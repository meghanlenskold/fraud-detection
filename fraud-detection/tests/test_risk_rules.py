from risk_rules import label_risk, score_transaction


BASE_TX = {
    "device_risk_score": 10,
    "is_international": 0,
    "amount_usd": 100,
    "velocity_24h": 1,
    "failed_logins_24h": 0,
    "prior_chargebacks": 0,
}


def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_large_amount_adds_risk():
    tx = {**BASE_TX, "amount_usd": 1200}
    assert score_transaction(tx) >= 25


def test_high_risk_device_increases_score():
    low_device = score_transaction({**BASE_TX, "device_risk_score": 10})
    high_device = score_transaction({**BASE_TX, "device_risk_score": 75})
    assert high_device > low_device


def test_international_increases_score():
    domestic = score_transaction({**BASE_TX, "is_international": 0})
    international = score_transaction({**BASE_TX, "is_international": 1})
    assert international > domestic


def test_high_velocity_increases_score():
    low_velocity = score_transaction({**BASE_TX, "velocity_24h": 1})
    high_velocity = score_transaction({**BASE_TX, "velocity_24h": 6})
    assert high_velocity > low_velocity


def test_prior_chargebacks_increase_score():
    no_cb = score_transaction({**BASE_TX, "prior_chargebacks": 0})
    one_cb = score_transaction({**BASE_TX, "prior_chargebacks": 1})
    two_cb = score_transaction({**BASE_TX, "prior_chargebacks": 2})
    assert one_cb > no_cb
    assert two_cb > one_cb


def test_worst_case_transaction_is_high_risk():
    # All fraud signals present simultaneously — must score high
    tx = {
        "device_risk_score": 85,
        "is_international": 1,
        "amount_usd": 1400,
        "velocity_24h": 8,
        "failed_logins_24h": 7,
        "prior_chargebacks": 1,
    }
    assert label_risk(score_transaction(tx)) == "high"
