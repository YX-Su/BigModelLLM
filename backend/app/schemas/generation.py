"""Data models for the generation and rule-validation stage."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    """LLM generation output. ``config_plan`` is populated for plan-producing intents."""

    intent: str
    answer: str
    config_plan: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    """A single problem found while checking a generated config plan."""

    rule_id: str = ""
    severity: str = "error"  # "error" | "warning"
    message: str


class ValidationReport(BaseModel):
    """Result of validating a config plan against dependency / mutex rules."""

    passed: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)
    suggested_plan: list[str] = Field(default_factory=list)

    def render_note(self) -> str:
        """Render the validation outcome as a note appended to the answer."""
        if not self.issues:
            return "配置校验:方案通过依赖与互斥规则校验。"
        lines = ["配置校验发现以下问题:"]
        for issue in self.issues:
            tag = f"[{issue.rule_id}] " if issue.rule_id else ""
            mark = "错误" if issue.severity == "error" else "提示"
            lines.append(f"- ({mark}) {tag}{issue.message}")
        if self.suggested_plan:
            lines.append("修正后建议方案:" + " + ".join(self.suggested_plan))
        return "\n".join(lines)
