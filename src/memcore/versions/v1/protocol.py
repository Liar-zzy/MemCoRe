"""Execution contracts independent of persisted graph construction."""

# Version the read/evaluation behavior separately so old QA checkpoints cannot
# silently mix old query parsing or evidence budgets with a new QA run.
EXECUTION_REVISION = "qa-round-budget-subquery-v1"
