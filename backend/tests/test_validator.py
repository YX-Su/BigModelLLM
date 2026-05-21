"""Unit tests for rule-system plan validation (hand-built RuleSet, no Neo4j)."""
from __future__ import annotations

from app.generation.validator import RuleSet, validate
from app.schemas.knowledge import Relation


def _ruleset() -> RuleSet:
    return RuleSet(
        mutex=[
            Relation(source="RP-CONSERVATIVE", target="STRAT-LEVERAGE", type="MUTEX", rule_id="R-MEX-05"),
        ],
        depends=[
            Relation(source="STRAT-LEVERAGE", target="RP-AGGRESSIVE", type="DEPENDS_ON", rule_id="R-DEP-05"),
            Relation(source="RISK-TAKEPROFIT", target="ASSET-EQUITY", type="DEPENDS_ON", rule_id="R-DEP-03"),
        ],
        known_codes={
            "RP-CONSERVATIVE", "RP-AGGRESSIVE", "STRAT-LEVERAGE",
            "ASSET-EQUITY", "RISK-TAKEPROFIT", "ASSET-BOND",
        },
    )


def test_clean_plan_passes():
    report = validate(["RP-AGGRESSIVE", "ASSET-EQUITY", "RISK-TAKEPROFIT"], _ruleset())
    assert report.passed
    assert not report.issues


def test_mutex_conflict_detected_and_corrected():
    report = validate(["RP-CONSERVATIVE", "STRAT-LEVERAGE"], _ruleset())
    assert not report.passed
    assert any(i.rule_id == "R-MEX-05" for i in report.issues)
    assert "STRAT-LEVERAGE" not in report.suggested_plan


def test_missing_dependency_detected_and_corrected():
    report = validate(["RISK-TAKEPROFIT", "ASSET-BOND"], _ruleset())
    assert not report.passed
    assert any(i.rule_id == "R-DEP-03" for i in report.issues)
    assert "ASSET-EQUITY" in report.suggested_plan


def test_unknown_code_is_warning_not_error():
    report = validate(["RP-AGGRESSIVE", "ASSET-FOOBAR"], _ruleset())
    assert report.passed  # warning only
    assert any(i.severity == "warning" for i in report.issues)
