"""Evaluation harness.

Implements the two-layer metric system:

* **Process metrics** — intent recognition accuracy, RAG recall (with vs.
  without the knowledge graph), and generation task accuracy.
* **Result metrics** — business-facing outcomes are defined in
  ``docs/evaluation.md``; only the process metrics are computed automatically.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from app.config import DATA_DIR
from app.intent.classifier import classify
from app.rag.chunking import extract_config_codes
from app.schemas.retrieval import ScoredChunk

logger = logging.getLogger(__name__)

EVAL_DIR = DATA_DIR / "eval"


def load_dataset(name: str) -> list[dict[str, Any]]:
    """Load an evaluation test set from ``data/eval/``."""
    path: Path = EVAL_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _covered_codes(chunks: list[ScoredChunk]) -> set[str]:
    """Union of config item codes mentioned across a set of retrieved chunks."""
    codes: set[str] = set()
    for chunk in chunks:
        codes.update(extract_config_codes(chunk.text))
    return codes


def evaluate_intent() -> dict[str, Any]:
    """Process metric — intent recognition accuracy."""
    dataset = load_dataset("intent_testset.json")
    correct = 0
    mistakes: list[dict[str, str]] = []
    for case in dataset:
        predicted = classify(case["query"]).intent
        if predicted == case["expected_intent"]:
            correct += 1
        else:
            mistakes.append(
                {
                    "query": case["query"],
                    "expected": case["expected_intent"],
                    "predicted": predicted,
                }
            )
    return {
        "metric": "意图识别准确率",
        "total": len(dataset),
        "correct": correct,
        "accuracy": correct / len(dataset) if dataset else 0.0,
        "mistakes": mistakes,
    }


def evaluate_retrieval() -> dict[str, Any]:
    """Process metric — RAG recall, comparing baseline vs. GraphRAG."""
    # Imported lazily: needs Milvus / Elasticsearch / Neo4j to be running.
    from app.rag.retrieval import retrieve, retrieve_baseline

    dataset = load_dataset("retrieval_testset.json")
    baseline_recalls: list[float] = []
    graphrag_recalls: list[float] = []
    for case in dataset:
        gold = set(case["gold_codes"])
        if not gold:
            continue
        baseline = _covered_codes(retrieve_baseline(case["query"]))
        graphrag = _covered_codes(retrieve(case["query"]).chunks)
        baseline_recalls.append(len(gold & baseline) / len(gold))
        graphrag_recalls.append(len(gold & graphrag) / len(gold))
    return {
        "metric": "RAG 检索召回率",
        "total": len(dataset),
        "baseline_recall": _mean(baseline_recalls),
        "graphrag_recall": _mean(graphrag_recalls),
    }


def evaluate_generation() -> dict[str, Any]:
    """Process metric — generation task accuracy (plan correctness + rule pass)."""
    # Imported lazily: needs the full storage stack and an LLM endpoint.
    from app.pipeline import run_pipeline

    dataset = load_dataset("generation_testset.json")
    passed = 0
    failures: list[dict[str, Any]] = []
    for case in dataset:
        session_id = f"eval-{uuid.uuid4().hex[:8]}"
        result = run_pipeline(session_id, case["query"])
        plan = set(result.config_plan)
        must_ok = set(case.get("must_codes", [])) <= plan
        forbidden_ok = not (set(case.get("forbidden_codes", [])) & plan)
        rule_ok = result.validation is None or result.validation.passed
        if must_ok and forbidden_ok and rule_ok:
            passed += 1
        else:
            failures.append(
                {
                    "query": case["query"],
                    "plan": result.config_plan,
                    "must_ok": must_ok,
                    "forbidden_ok": forbidden_ok,
                    "rule_ok": rule_ok,
                }
            )
    return {
        "metric": "大模型生成任务准确率",
        "total": len(dataset),
        "passed": passed,
        "accuracy": passed / len(dataset) if dataset else 0.0,
        "failures": failures,
    }
