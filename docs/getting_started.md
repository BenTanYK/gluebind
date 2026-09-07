# Getting started

## Grid Engine

Keep scheduler settings in a cluster-owned YAML file, separate from a portable
calculation ``config.yaml``. Start with ``gluebind/data/grid_engine_config.yaml``
and customise it for the local queue and software environment. For example, an
Eddie-style preamble can initialise modules and the calculation environment:

```yaml
queue: gpu
time: "24:00:00"
memory: 24G
resources: {gpu: "1"}
preamble:
  - . /etc/profile.d/modules.sh
  - module load cuda
  - source /home/USER/miniconda3/etc/profile.d/conda.sh
  - conda activate gluebind
```

Submit a calculation with the same configuration used to construct the backend:

```python
from gluebind import Calculation, GridEngineConfig
from gluebind.backend import GridEngineBackend

grid_engine = GridEngineConfig.load("grid_engine_config.yaml")
calc = Calculation.from_config(
    "config.yaml",
    GridEngineBackend(grid_engine),
    scheduler_config=grid_engine,
)
calc.run()
```

GlueBind writes a script in each job's working directory and invokes ``qsub``
from that directory, so Grid Engine's ``#$ -cwd`` directive selects the correct
workspace. Completion still requires the expected GlueBind result artifact on
the shared filesystem; leaving ``qstat`` alone is not treated as success.
