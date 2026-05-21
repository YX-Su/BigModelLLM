#!/usr/bin/env python3
"""CLI for the GraphRAG portfolio assistant evaluation harness.

Usage:
    python scripts/evaluate.py --intent        # intent accuracy only (no services)
    python scripts/evaluate.py --retrieval     # RAG recall (needs storage stack)
    python scripts/evaluate.py --generation    # generation accuracy (needs stack + LLM)
    python scripts/evaluate.py --all           # everything available
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.eval.metrics import (  # noqa: E402
    evaluate_generation,
    evaluate_intent,
    evaluate_retrieval,
)


def _print_intent() -> None:
    report = evaluate_intent()
    print(f"\n[过程指标] {report['metric']}")
    print(f"  样本数: {report['total']}  正确: {report['correct']}")
    print(f"  准确率: {report['accuracy']:.1%}")
    for miss in report["mistakes"]:
        print(f"  ✗ {miss['query']} -> 预测 {miss['predicted']} / 期望 {miss['expected']}")


def _print_retrieval() -> None:
    report = evaluate_retrieval()
    base = report["baseline_recall"]
    graph = report["graphrag_recall"]
    print(f"\n[过程指标] {report['metric']}  (样本数: {report['total']})")
    print(f"  纯向量基线召回率(无图谱): {base:.1%}")
    print(f"  GraphRAG 召回率(有图谱): {graph:.1%}")
    print(f"  图谱带来的召回提升: {graph - base:+.1%}")


def _print_generation() -> None:
    report = evaluate_generation()
    print(f"\n[过程指标] {report['metric']}  (样本数: {report['total']})")
    print(f"  通过: {report['passed']}  准确率: {report['accuracy']:.1%}")
    for fail in report["failures"]:
        print(f"  ✗ {fail['query']}")
        print(f"    plan={fail['plan']} must={fail['must_ok']} "
              f"forbidden={fail['forbidden_ok']} rule={fail['rule_ok']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG 投资组合配置助手测评")
    parser.add_argument("--intent", action="store_true", help="意图识别准确率")
    parser.add_argument("--retrieval", action="store_true", help="RAG 召回率(需存储栈)")
    parser.add_argument("--generation", action="store_true", help="生成准确率(需存储栈+LLM)")
    parser.add_argument("--all", action="store_true", help="运行全部可用测评")
    args = parser.parse_args()

    if not any([args.intent, args.retrieval, args.generation, args.all]):
        args.all = True

    logging.basicConfig(level="WARNING")
    print("=" * 56)
    print(" GraphRAG 投资组合配置助手 — 测评报告")
    print("=" * 56)

    if args.intent or args.all:
        _print_intent()
    if args.retrieval or args.all:
        try:
            _print_retrieval()
        except Exception as exc:  # noqa: BLE001
            print(f"\n[过程指标] RAG 检索召回率: 跳过 ({exc})")
    if args.generation or args.all:
        try:
            _print_generation()
        except Exception as exc:  # noqa: BLE001
            print(f"\n[过程指标] 生成准确率: 跳过 ({exc})")

    print("\n" + "=" * 56)
    print(" 结果指标(业务落地)请参见 docs/evaluation.md")
    print("=" * 56)


if __name__ == "__main__":
    main()
