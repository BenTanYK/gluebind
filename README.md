<h1 align="center">gluebind</h1>

<p align="center">Automated Umbrella Sampling protocol for calculating the binding free energy of ternary complexes</p>

<p align="center">
  <a href="https://github.com/BenTanYK/gluebind/actions?query=workflow%3Aci">
    <img alt="ci" src="https://github.com/BenTanYK/gluebind/actions/workflows/ci.yaml/badge.svg" />
  </a>
  <a href="https://app.codacy.com/gh/BenTanYK/gluebind/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade">
    <img alt="Codacy Badge" src="https://app.codacy.com/project/badge/Grade/ee01368a0c6341999635699ca57d7edf" />
  </a>
  <a href="https://codecov.io/gh/BenTanYK/gluebind/branch/main">
    <img alt="coverage" src="https://codecov.io/gh/BenTanYK/gluebind/branch/main/graph/badge.svg" />
  </a>
  <a href="https://opensource.org/license/gpl-3.0">
    <img alt="license" src="https://img.shields.io/badge/License-GPL%20v3-blue.svg" />
  </a>
  <a href="https://github.com/astral-sh/ty">
    <img alt="Checked with ty" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json" />
  </a>
  <a href="https://www.mdanalysis.org">
    <img alt="Powered by MDAnalysis" title="Powered by MDAnalysis" src="https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA">
  </a>
</p>

---

> [!CAUTION]
> The `gluebind` package is still experimental and untested - not ready for production use yet!

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

### Acknowledgements

Project based on the Simon Boothroyd's excellent [cookiecutter](https://github.com/SimonBoothroyd/python-template/tree/main), with help from [Finlay Clark](https://github.com/fjclark)'s [useful tutorial](https://github.com/fjclark/python_package_tutorial).
