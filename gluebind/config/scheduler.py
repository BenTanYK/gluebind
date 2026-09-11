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
