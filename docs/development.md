# Development

## Scheduler backends

Cluster backends implement ``submit``, ``poll``, and ``cancel`` from
``gluebind.backend.Backend``. They receive scheduler-neutral ``JobSpec`` objects;
worker code must not acquire scheduler-specific branches. Detached backends must
preserve opaque handles, account for the short interval before a submitted job
appears in the queue, and leave final success validation to the caller's
on-disk result-artifact check.

Grid Engine changes should be unit-tested with mocked ``qsub``, ``qstat``, and
``qdel``, then smoke-tested on the target cluster with a real submit, poll, and
cancel cycle. Scheduler configuration is cluster-scoped; do not put queues,
modules, accounts, or user environment paths into a calculation config or
package source.

To create a development environment, you must have [`mamba` installed](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

A development conda environment can be created and activated with:

```shell
make env
conda activate gluebind
```

To format the codebase:

```shell
make format
```

To run the unit tests:

```shell
make test
```

To serve the documentation locally:

```shell
mkdocs serve
```
