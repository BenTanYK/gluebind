"""Common, cluster-scoped settings used by scheduler backends."""

from __future__ import annotations

import pydantic


class SchedulerConfig(pydantic.BaseModel):
    """Settings consumed by GlueBind's backend-neutral job scheduler."""

    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    queue_check_interval: int = pydantic.Field(
        30, ge=1, description="Seconds between scheduler queue polls."
    )
    job_submission_wait: int = pydantic.Field(
        300,
        ge=1,
        description="Seconds to wait for a submitted job to appear in the queue.",
    )
    queue_len_lim: int = pydantic.Field(
        2000,
        ge=1,
        description="Maximum number of jobs allowed in the scheduler queue at once.",
    )
    poll_retries: int = pydantic.Field(
        5,
        ge=1,
        description=(
            "Attempts at a queue-status query (squeue/qstat) before giving up, so "
            "a transient controller error does not stop the driver."
        ),
    )
    poll_retry_wait_s: float = pydantic.Field(
        30.0,
        gt=0,
        description=(
            "Seconds to wait before the first retry of a failed queue-status query; "
            "each further retry waits twice as long."
        ),
    )
