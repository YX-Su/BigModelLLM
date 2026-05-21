"""Rule-system backstop validation for generated config plans.

The LLM may produce a portfolio plan that violates business constraints. This
module checks a plan against the dependency and mutex rules held in the
knowledge graph, and proposes a corrected plan where possible — so plans that
break business constraints never reach the user unflagged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from app.graph.store import GraphStore, get_graph_store
from app.schemas.generation import ValidationIssue, ValidationReport
from app.schemas.knowledge import Relation

logger = logging.getLogger(__name__)


@dataclass
class RuleSet:
    """Dependency / mutex rules plus the set of known config item codes."""

    mutex: list[Relation] = field(default_factory=list)
    depends: list[Relation] = field(default_factory=list)
    known_codes: set[str] = field(default_factory=set)


def load_ruleset(store: GraphStore | None = None) -> RuleSet:
    """Load the rule set from the knowledge graph."""
    store = store or get_graph_store()
    return RuleSet(
        mutex=store.all_relations("MUTEX"),
        depends=store.all_relations("DEPENDS_ON"),
        known_codes=store.config_item_ids(),
    )


@lru_cache
def cached_ruleset() -> RuleSet:
    """Process-cached rule set (the graph is static after the offline build)."""
    return load_ruleset()


def validate(plan: list[str], ruleset: RuleSet) -> ValidationReport:
    """Check a config plan against dependency, mutex and code-validity rules."""
    issues: list[ValidationIssue] = []
    plan_set = set(plan)
    suggested = list(plan)

    # Code validity — every code must be a known config item.
    for code in plan:
        if ruleset.known_codes and code not in ruleset.known_codes:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"配置项「{code}」不在已知配置项清单中,请核对编码",
                )
            )

    # Mutex — two mutually exclusive items must not both be enabled.
    seen_pairs: set[frozenset[str]] = set()
    for rel in ruleset.mutex:
        pair = frozenset({rel.source, rel.target})
        if len(pair) != 2 or pair in seen_pairs:
            continue
        if rel.source in plan_set and rel.target in plan_set:
            seen_pairs.add(pair)
            if rel.source in suggested and rel.target in suggested:
                later = (
                    rel.source
                    if suggested.index(rel.source) > suggested.index(rel.target)
                    else rel.target
                )
            else:
                later = rel.target
            issues.append(
                ValidationIssue(
                    rule_id=rel.rule_id,
                    severity="error",
                    message=(
                        f"配置项「{rel.source}」与「{rel.target}」互斥,不能同时开启;"
                        f"建议移除「{later}」"
                    ),
                )
            )
            if later in suggested:
                suggested.remove(later)

    # Dependency — at least one DEPENDS_ON target must be present (OR semantics).
    deps: dict[str, list[Relation]] = {}
    for rel in ruleset.depends:
        deps.setdefault(rel.source, []).append(rel)
    for code in plan:
        rels = deps.get(code)
        if not rels:
            continue
        targets = [r.target for r in rels]
        if not any(target in plan_set for target in targets):
            issues.append(
                ValidationIssue(
                    rule_id=rels[0].rule_id,
                    severity="error",
                    message=(
                        f"配置项「{code}」依赖 {' 或 '.join(targets)},"
                        f"当前方案均未开启;建议补充「{targets[0]}」"
                    ),
                )
            )
            if targets[0] not in suggested:
                suggested.append(targets[0])

    passed = not any(issue.severity == "error" for issue in issues)
    if not passed:
        logger.info("plan validation failed: %d issue(s)", len(issues))
    return ValidationReport(passed=passed, issues=issues, suggested_plan=suggested)
