<h1 align="center">gluebind</h1>

<p align="center">Automated Umbrella Sampling protocol for calculating the binding free energy of ternary complexes</p>

<p align="center">
  <a href="https://github.com/BenTanYK/gluebind/actions?query=workflow%3Aci">
    <img alt="ci" src="https://github.com/BenTanYK/gluebind/actions/workflows/ci.yaml/badge.svg" />
  </a>
  <a href="https://codecov.io/gh/BenTanYK/gluebind/branch/main">
    <img alt="coverage" src="https://codecov.io/gh/BenTanYK/gluebind/branch/main/graph/badge.svg" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="license" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

---

The `gluebind` framework ...

## Installation

`gluebind` depends on [BioSimSpace](https://biosimspace.openbiosim.org/) (and the
AmberTools / GROMACS backends it drives), which are distributed via conda rather
than PyPI. It is therefore installed from source into a conda environment, so
you'll need [`mamba`](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
(or `conda`):

```shell
git clone https://github.com/BenTanYK/gluebind.git
cd gluebind
make env
conda activate gluebind
```

`make env` creates a conda environment named `gluebind` from
`devtools/envs/test.yaml` — pulling BioSimSpace, OpenMM, AmberTools, GROMACS and
the rest from the `conda-forge` and `openbiosim` channels — and installs
`gluebind` into it in editable mode.

### WHAM (analysis prerequisite)

The free-energy analysis stage uses Grossfield
[WHAM](https://github.com/agrossfield/wham), which is not available through
conda. With the `gluebind` environment active (`conda activate gluebind`), clone
and compile it with:

```shell
make wham
```

This clones Grossfield WHAM from GitHub, compiles it, and installs the `wham`
binary into the active environment's `bin/` — so it is on your `PATH` and
gluebind finds it by default. If the pinned release is unavailable, or you are on
a machine without network access, use one of the fallbacks:

```shell
# A different release or branch
make wham WHAM_REF=<tag-or-branch>

# A fork or mirror
make wham WHAM_REPO=<url>

# Build from an existing local checkout (no clone)
make wham WHAM_SRC=/path/to/wham
```

As a last resort you can compile WHAM by hand and place the `wham` binary
anywhere on your `PATH`; gluebind resolves it the same way.

## Getting Started

To get started, see the [documentation](https://BenTanYK.github.io/gluebind/latest/).
