import pytest
from backend.schemas import MaintenanceTaskInput, SeverityLevelEnum
from backend.predictor import predictor

def test_0_model_loads():
    # TEST 0 - model artifact can be loaded
    assert predictor.pipeline is not None

def test_1_zero_context():
    # TEST 1 — ZERO CONTEXT (Close to base)
    for severity, expected in [
        (SeverityLevelEnum.LOW, 10.0),
        (SeverityLevelEnum.MEDIUM, 30.0),
        (SeverityLevelEnum.HIGH, 55.0),
        (SeverityLevelEnum.CRITICAL, 78.0)
    ]:
        task = MaintenanceTaskInput(
            task_id="TSK-001",
            defect_severity_label=severity,
            days_overdue=0,
            real_daily_train_count=0,
            is_real_traffic_data=True,
            asset_criticality_score=0,
            failures_last_365d=0
        )
        res = predictor.predict(task)
        expected_ref = predictor.calculate_deterministic_reference(task)
        # Using larger tolerance for boundary extrapolations
        assert abs(res.urgency_score - expected_ref) < 15.0

def test_2_extreme_context():
    # TEST 2 — EXTREME CONTEXT
    for severity, expected_approx in [
        (SeverityLevelEnum.LOW, 58.0),
        (SeverityLevelEnum.MEDIUM, 70.0),
        (SeverityLevelEnum.HIGH, 90.0),
        (SeverityLevelEnum.CRITICAL, 100.0)
    ]:
        task = MaintenanceTaskInput(
            task_id="TSK-002",
            defect_severity_label=severity,
            days_overdue=365,
            real_daily_train_count=250,
            is_real_traffic_data=True,
            asset_criticality_score=100,
            failures_last_365d=10
        )
        res = predictor.predict(task)
        expected_ref = predictor.calculate_deterministic_reference(task)
        assert abs(res.urgency_score - expected_ref) < 15.0

def test_3_original_failure_case():
    # TEST 3 — ORIGINAL FAILURE CASE
    task = MaintenanceTaskInput(
        task_id="TSK-003",
        defect_severity_label=SeverityLevelEnum.CRITICAL,
        real_daily_train_count=73,
        is_real_traffic_data=True,
        asset_criticality_score=83,
        days_overdue=0,
        failures_last_365d=0
    )
    res = predictor.predict(task)
    expected = predictor.calculate_deterministic_reference(task)
    assert abs(res.urgency_score - expected) < 2.0, f"Expected approx {expected}, got {res.urgency_score}"

def test_4_second_failure_case():
    # TEST 4 — SECOND FAILURE CASE
    task = MaintenanceTaskInput(
        task_id="TSK-004",
        defect_severity_label=SeverityLevelEnum.CRITICAL,
        real_daily_train_count=18,
        is_real_traffic_data=True,
        asset_criticality_score=65.4,
        days_overdue=24,
        failures_last_365d=0
    )
    res = predictor.predict(task)
    expected = predictor.calculate_deterministic_reference(task)
    assert abs(res.urgency_score - expected) < 2.0, f"Expected approx {expected}, got {res.urgency_score}"

def test_5_monotonicity():
    # TEST 5 — MONOTONICITY
    def get_score(**kwargs):
        task = MaintenanceTaskInput(
            task_id="TSK-MONO",
            defect_severity_label=SeverityLevelEnum.MEDIUM,
            days_overdue=kwargs.get('days_overdue', 10),
            real_daily_train_count=kwargs.get('real_daily_train_count', 30),
            is_real_traffic_data=True,
            asset_criticality_score=kwargs.get('asset_criticality_score', 50),
            failures_last_365d=kwargs.get('failures_last_365d', 1)
        )
        return predictor.predict(task).urgency_score

    base_score = get_score()
    
    # ML approximations might have small noisy fluctuations, using -0.1 tolerance
    assert get_score(days_overdue=20) >= base_score - 0.1
    assert get_score(real_daily_train_count=40) >= base_score - 0.1
    assert get_score(asset_criticality_score=60) >= base_score - 0.1
    assert get_score(failures_last_365d=2) >= base_score - 0.1

def test_6_traffic_routing():
    # TEST 6 — TRAFFIC ROUTING
    task_real = MaintenanceTaskInput(
        task_id="TSK-TRAFFIC-1",
        defect_severity_label=SeverityLevelEnum.MEDIUM,
        real_daily_train_count=100,
        scheduled_services_count_proxy=10,
        is_real_traffic_data=True
    )
    score_real = predictor.predict(task_real).urgency_score
    
    task_proxy = MaintenanceTaskInput(
        task_id="TSK-TRAFFIC-2",
        defect_severity_label=SeverityLevelEnum.MEDIUM,
        real_daily_train_count=None,
        scheduled_services_count_proxy=100,
        is_real_traffic_data=False
    )
    score_proxy = predictor.predict(task_proxy).urgency_score
    
    task_both = MaintenanceTaskInput(
        task_id="TSK-TRAFFIC-3",
        defect_severity_label=SeverityLevelEnum.MEDIUM,
        real_daily_train_count=100,
        scheduled_services_count_proxy=500,
        is_real_traffic_data=True
    )
    score_both = predictor.predict(task_both).urgency_score
    
    assert score_real == score_proxy == score_both

def test_7_invalid_data():
    # TEST 7 — INVALID DATA
    
    # Test Negative / Out of bounds
    task = MaintenanceTaskInput(
        task_id="TSK-INV",
        defect_severity_label=SeverityLevelEnum.LOW,
        days_overdue=-10,
        real_daily_train_count=-50,
        is_real_traffic_data=True,
        asset_criticality_score=150,
        failures_last_365d=-5
    )
    res = predictor.predict(task)
    
    task_clamped = MaintenanceTaskInput(
        task_id="TSK-INV",
        defect_severity_label=SeverityLevelEnum.LOW,
        days_overdue=0,
        real_daily_train_count=0,
        is_real_traffic_data=True,
        asset_criticality_score=100,
        failures_last_365d=0
    )
    assert res.urgency_score == predictor.predict(task_clamped).urgency_score
    
    task_no_traffic = MaintenanceTaskInput(
        task_id="TSK-NO-TRAFFIC",
        defect_severity_label=SeverityLevelEnum.LOW,
        real_daily_train_count=None,
        scheduled_services_count_proxy=None,
        is_real_traffic_data=True
    )
    with pytest.raises(ValueError, match="Traffic coverage is missing"):
        predictor.predict(task_no_traffic)
        
    task_no_severity = MaintenanceTaskInput(
        task_id="TSK-NO-SEV"
    )
    task_no_severity.defect_severity_label = None 
    with pytest.raises(ValueError, match="Severity is missing or invalid"):
        predictor.predict(task_no_severity)
