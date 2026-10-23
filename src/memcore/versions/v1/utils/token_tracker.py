"""E1: Token and latency tracking for LLM calls.

Usage:
    from memcore.versions.v1.utils.token_tracker import get_construction_tracker, get_retrieval_tracker

    # In OpenAIController.get_completion():
    tracker.record("caller_name", input_tokens, output_tokens, wall_time)

    # Get summary:
    stats = tracker.get_summary()
    tracker.reset()
"""
import time
import json
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class LLMCallRecord:
    caller: str
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_s: float = 0.0
    model: str = ""


class TokenTracker:
    """Thread-safe token and latency tracker."""

    def __init__(self):
        self._records: List[LLMCallRecord] = []
        self._lock = threading.Lock()
        self.enabled = False

    def record(self, caller: str, input_tokens: int, output_tokens: int,
               wall_time_s: float, model: str = ""):
        if not self.enabled:
            return
        with self._lock:
            self._records.append(LLMCallRecord(
                caller=caller,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                wall_time_s=wall_time_s,
                model=model,
            ))

    def get_summary(self) -> Dict:
        with self._lock:
            total_input = sum(r.input_tokens for r in self._records)
            total_output = sum(r.output_tokens for r in self._records)
            total_time = sum(r.wall_time_s for r in self._records)
            count = len(self._records)

            by_caller = {}
            for r in self._records:
                if r.caller not in by_caller:
                    by_caller[r.caller] = {
                        "input_tokens": 0, "output_tokens": 0,
                        "wall_time_s": 0.0, "count": 0
                    }
                by_caller[r.caller]["input_tokens"] += r.input_tokens
                by_caller[r.caller]["output_tokens"] += r.output_tokens
                by_caller[r.caller]["wall_time_s"] += r.wall_time_s
                by_caller[r.caller]["count"] += 1

            return {
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "total_wall_time_s": round(total_time, 3),
                "call_count": count,
                "by_caller": by_caller,
            }

    def reset(self):
        with self._lock:
            self._records.clear()

    def save(self, path: str):
        summary = self.get_summary()
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)


# Global singleton trackers
_construction_tracker = TokenTracker()
_retrieval_tracker = TokenTracker()


def get_construction_tracker() -> TokenTracker:
    return _construction_tracker


def get_retrieval_tracker() -> TokenTracker:
    return _retrieval_tracker


def enable_tracking():
    _construction_tracker.enabled = True
    _retrieval_tracker.enabled = True


def disable_tracking():
    _construction_tracker.enabled = False
    _retrieval_tracker.enabled = False
