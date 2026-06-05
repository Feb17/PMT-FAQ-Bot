"""Alert threshold monitoring for retrieval quality and ingest health.

Defines configurable thresholds and check functions that emit
structured warning/critical alerts via Python logging.
Thresholds are overridable via environment variables.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold definitions
# ---------------------------------------------------------------------------


@dataclass
class Thresholds:
    """Configurable alert thresholds with env-var overrides."""

    # --- Retrieval quality ---
    pass_rate_warning_pct: float = field(
        default_factory=lambda: float(
            os.getenv("ALERT_PASS_RATE_WARNING_PCT", "85.0")
        )
    )
    pass_rate_critical_pct: float = field(
        default_factory=lambda: float(
            os.getenv("ALERT_PASS_RATE_CRITICAL_PCT", "70.0")
        )
    )
    pass_rate_drop_warning_pct: float = field(
        default_factory=lambda: float(
            os.getenv("ALERT_PASS_RATE_DROP_WARNING_PCT", "10.0")
        )
    )
    pass_rate_drop_critical_pct: float = field(
        default_factory=lambda: float(
            os.getenv("ALERT_PASS_RATE_DROP_CRITICAL_PCT", "20.0")
        )
    )

    # --- Rerank score degradation ---
    avg_rerank_score_warning: float = field(
        default_factory=lambda: float(
            os.getenv("ALERT_AVG_RERANK_SCORE_WARNING", "0.5")
        )
    )
    rerank_score_drop_warning_pct: float = field(
        default_factory=lambda: float(
            os.getenv("ALERT_RERANK_SCORE_DROP_WARNING_PCT", "15.0")
        )
    )

    # --- No-match accuracy ---
    nomatch_accuracy_warning_pct: float = field(
        default_factory=lambda: float(
            os.getenv("ALERT_NOMATCH_ACCURACY_WARNING_PCT", "80.0")
        )
    )

    # --- Ingest ---
    ingest_failure_critical: int = field(
        default_factory=lambda: int(
            os.getenv("ALERT_INGEST_FAILURE_CRITICAL", "1")
        )
    )
    ingest_duration_warning_seconds: float = field(
        default_factory=lambda: float(
            os.getenv("ALERT_INGEST_DURATION_WARNING_SECONDS", "600.0")
        )
    )

    # --- Latency (milliseconds) ---
    retrieval_p95_warning_ms: float = field(
        default_factory=lambda: float(
            os.getenv("ALERT_RETRIEVAL_P95_WARNING_MS", "5000.0")
        )
    )
    retrieval_p95_critical_ms: float = field(
        default_factory=lambda: float(
            os.getenv("ALERT_RETRIEVAL_P95_CRITICAL_MS", "10000.0")
        )
    )

    # --- Collection health ---
    points_count_drop_warning_pct: float = field(
        default_factory=lambda: float(
            os.getenv("ALERT_POINTS_COUNT_DROP_WARNING_PCT", "20.0")
        )
    )


# ---------------------------------------------------------------------------
# Alert record
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    level: str       # "warning" or "critical"
    category: str    # "retrieval_quality", "ingest", "latency", "collection_health"
    message: str
    value: float
    threshold: float
    baseline_value: Optional[float] = None


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_pass_rate(
    current_pass_rate: float,
    baseline_pass_rate: Optional[float] = None,
    thresholds: Optional[Thresholds] = None,
) -> list[Alert]:
    """Check retrieval pass rate against absolute and regression thresholds."""
    t = thresholds or Thresholds()
    alerts: list[Alert] = []

    # Absolute thresholds
    if current_pass_rate < t.pass_rate_critical_pct:
        alerts.append(
            Alert(
                "critical",
                "retrieval_quality",
                f"Pass rate {current_pass_rate:.1f}% below critical "
                f"threshold {t.pass_rate_critical_pct:.1f}%",
                current_pass_rate,
                t.pass_rate_critical_pct,
            )
        )
    elif current_pass_rate < t.pass_rate_warning_pct:
        alerts.append(
            Alert(
                "warning",
                "retrieval_quality",
                f"Pass rate {current_pass_rate:.1f}% below warning "
                f"threshold {t.pass_rate_warning_pct:.1f}%",
                current_pass_rate,
                t.pass_rate_warning_pct,
            )
        )

    # Regression from baseline
    if baseline_pass_rate is not None:
        drop = baseline_pass_rate - current_pass_rate
        if drop > t.pass_rate_drop_critical_pct:
            alerts.append(
                Alert(
                    "critical",
                    "retrieval_quality",
                    f"Pass rate dropped {drop:.1f}% from baseline "
                    f"{baseline_pass_rate:.1f}%",
                    drop,
                    t.pass_rate_drop_critical_pct,
                    baseline_pass_rate,
                )
            )
        elif drop > t.pass_rate_drop_warning_pct:
            alerts.append(
                Alert(
                    "warning",
                    "retrieval_quality",
                    f"Pass rate dropped {drop:.1f}% from baseline "
                    f"{baseline_pass_rate:.1f}%",
                    drop,
                    t.pass_rate_drop_warning_pct,
                    baseline_pass_rate,
                )
            )

    return alerts


def check_rerank_score(
    current_avg: float,
    baseline_avg: Optional[float] = None,
    thresholds: Optional[Thresholds] = None,
) -> list[Alert]:
    """Check average rerank score against absolute and regression thresholds."""
    t = thresholds or Thresholds()
    alerts: list[Alert] = []

    if current_avg < t.avg_rerank_score_warning:
        alerts.append(
            Alert(
                "warning",
                "retrieval_quality",
                f"Average rerank score {current_avg:.3f} below "
                f"{t.avg_rerank_score_warning:.3f}",
                current_avg,
                t.avg_rerank_score_warning,
            )
        )

    if baseline_avg is not None and baseline_avg > 0:
        drop_pct = ((baseline_avg - current_avg) / baseline_avg) * 100.0
        if drop_pct > t.rerank_score_drop_warning_pct:
            alerts.append(
                Alert(
                    "warning",
                    "retrieval_quality",
                    f"Avg rerank score dropped {drop_pct:.1f}% from baseline "
                    f"{baseline_avg:.3f}",
                    drop_pct,
                    t.rerank_score_drop_warning_pct,
                    baseline_avg,
                )
            )

    return alerts


def check_nomatch_accuracy(
    nomatch_correct: int,
    nomatch_total: int,
    thresholds: Optional[Thresholds] = None,
) -> list[Alert]:
    """Check no-match rejection accuracy."""
    t = thresholds or Thresholds()
    alerts: list[Alert] = []
    if nomatch_total == 0:
        return alerts

    accuracy = (nomatch_correct / nomatch_total) * 100.0
    if accuracy < t.nomatch_accuracy_warning_pct:
        alerts.append(
            Alert(
                "warning",
                "retrieval_quality",
                f"No-match accuracy {accuracy:.0f}% ({nomatch_correct}/{nomatch_total}) "
                f"below {t.nomatch_accuracy_warning_pct:.0f}%",
                accuracy,
                t.nomatch_accuracy_warning_pct,
            )
        )
    return alerts


def check_ingest(
    errors: list[str],
    elapsed_seconds: float,
    thresholds: Optional[Thresholds] = None,
) -> list[Alert]:
    """Check ingest for errors and excessive duration."""
    t = thresholds or Thresholds()
    alerts: list[Alert] = []

    if len(errors) >= t.ingest_failure_critical:
        alerts.append(
            Alert(
                "critical",
                "ingest",
                f"Ingest had {len(errors)} error(s)",
                float(len(errors)),
                float(t.ingest_failure_critical),
            )
        )

    if elapsed_seconds > t.ingest_duration_warning_seconds:
        alerts.append(
            Alert(
                "warning",
                "ingest",
                f"Ingest took {elapsed_seconds:.0f}s "
                f"(>{t.ingest_duration_warning_seconds:.0f}s threshold)",
                elapsed_seconds,
                t.ingest_duration_warning_seconds,
            )
        )

    return alerts


def check_latency(
    p95_ms: float,
    thresholds: Optional[Thresholds] = None,
) -> list[Alert]:
    """Check retrieval p95 latency against thresholds."""
    t = thresholds or Thresholds()
    alerts: list[Alert] = []

    if p95_ms > t.retrieval_p95_critical_ms:
        alerts.append(
            Alert(
                "critical",
                "latency",
                f"Retrieval p95 latency {p95_ms:.0f}ms above critical "
                f"{t.retrieval_p95_critical_ms:.0f}ms",
                p95_ms,
                t.retrieval_p95_critical_ms,
            )
        )
    elif p95_ms > t.retrieval_p95_warning_ms:
        alerts.append(
            Alert(
                "warning",
                "latency",
                f"Retrieval p95 latency {p95_ms:.0f}ms above warning "
                f"{t.retrieval_p95_warning_ms:.0f}ms",
                p95_ms,
                t.retrieval_p95_warning_ms,
            )
        )

    return alerts


def check_collection_health(
    current_points: int,
    baseline_points: Optional[int] = None,
    thresholds: Optional[Thresholds] = None,
) -> list[Alert]:
    """Check collection points count for unexpected drops."""
    t = thresholds or Thresholds()
    alerts: list[Alert] = []

    if baseline_points is not None and baseline_points > 0:
        drop_pct = ((baseline_points - current_points) / baseline_points) * 100.0
        if drop_pct > t.points_count_drop_warning_pct:
            alerts.append(
                Alert(
                    "warning",
                    "collection_health",
                    f"Points count dropped {drop_pct:.1f}% "
                    f"({baseline_points} → {current_points})",
                    drop_pct,
                    t.points_count_drop_warning_pct,
                    float(baseline_points),
                )
            )

    return alerts


# ---------------------------------------------------------------------------
# Alert emitter
# ---------------------------------------------------------------------------


def emit_alerts(alerts: list[Alert]) -> int:
    """Log all alerts and return count of critical alerts.

    Uses structured log messages so external log processors / alerting
    tools can parse them.
    """
    critical_count = 0
    for a in alerts:
        if a.level == "critical":
            logger.error(
                "[ALERT:%s:%s] %s  value=%.3f threshold=%.3f",
                a.level.upper(),
                a.category,
                a.message,
                a.value,
                a.threshold,
            )
            critical_count += 1
        else:
            logger.warning(
                "[ALERT:%s:%s] %s  value=%.3f threshold=%.3f",
                a.level.upper(),
                a.category,
                a.message,
                a.value,
                a.threshold,
            )
    return critical_count
