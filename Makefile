PACKAGE_NAME := gluebind
PACKAGE_DIR  := gluebind

CONDA_ENV_RUN   = conda run --no-capture-output --name $(PACKAGE_NAME)

TEST_ARGS := -v --cov=$(PACKAGE_NAME) --cov-report=term --cov-report=xml --junitxml=unit.xml --color=yes

# Grossfield WHAM — cloned from GitHub + compiled into the active conda env by
# `make wham`. Override WHAM_REF for a different release/branch, WHAM_REPO for a
# fork/mirror, or WHAM_SRC to build from an existing local checkout (skips clone).
WHAM_REPO ?= https://github.com/agrossfield/wham.git
WHAM_REF  ?= v2.1.1
WHAM_SRC  ?=

.PHONY: env wham lint format test test-integration docs docs-deploy

env:
	mamba create     --name $(PACKAGE_NAME)
	mamba env update --name $(PACKAGE_NAME) --file devtools/envs/test.yaml
	$(CONDA_ENV_RUN) pip install --no-deps -e .
	$(CONDA_ENV_RUN) pre-commit install || true

# Clone + compile Grossfield WHAM into the active conda env's bin/ (on PATH, so
# gluebind's default `wham_binary="wham"` finds it). Layered fallbacks:
#   make wham                          # clone the pinned ref + build
#   make wham WHAM_REF=<tag-or-branch> # build a different release/branch
#   make wham WHAM_REPO=<url>          # clone a fork/mirror
#   make wham WHAM_SRC=/path/to/wham   # build from an existing local checkout
# If everything fails, compile by hand and drop `wham` on your PATH.
wham:
	@set -e; tmp=""; \
	if [ -n "$(WHAM_SRC)" ]; then \
	    src="$(WHAM_SRC)"; \
	    echo "Building WHAM from local checkout $$src"; \
	else \
	    tmp=$$(mktemp -d); src="$$tmp/wham"; \
	    if ! git clone --depth 1 --branch "$(WHAM_REF)" "$(WHAM_REPO)" "$$src"; then \
	        echo ""; \
	        echo "ERROR: could not clone WHAM ref '$(WHAM_REF)' from $(WHAM_REPO)"; \
	        echo "Fallback options:"; \
	        echo "  1. different release/branch:  make wham WHAM_REF=<tag-or-branch>"; \
	        echo "  2. fork or mirror:            make wham WHAM_REPO=<url>"; \
	        echo "  3. build from a local clone:  make wham WHAM_SRC=/path/to/wham"; \
	        echo "  4. or compile by hand and put 'wham' on your PATH (e.g. $$CONDA_PREFIX/bin)."; \
	        rm -rf "$$tmp"; exit 1; \
	    fi; \
	fi; \
	if [ ! -d "$$src/wham" ]; then echo "ERROR: no 'wham/' source dir under $$src"; [ -n "$$tmp" ] && rm -rf "$$tmp"; exit 1; fi; \
	$(MAKE) -C "$$src/wham"; \
	dest="$${CONDA_PREFIX:-/usr/local}/bin"; \
	mkdir -p "$$dest"; \
	cp "$$src/wham/wham" "$$dest/wham"; \
	echo "Installed wham -> $$dest/wham"; \
	[ -n "$$tmp" ] && rm -rf "$$tmp" || true

lint:
	$(CONDA_ENV_RUN) ruff check $(PACKAGE_DIR)

format:
	$(CONDA_ENV_RUN) ruff format $(PACKAGE_DIR)
	$(CONDA_ENV_RUN) ruff check --fix --select I $(PACKAGE_DIR)

test:
	$(CONDA_ENV_RUN) pytest -v $(TEST_ARGS) $(PACKAGE_DIR)/tests/

# The 1FAP integration tier (opt-in; overrides the default -m). Tests self-skip
# when a dep (BSS, red, wham) or a GPU is missing, so a partial env runs a partial
# tier rather than failing. Runs via LocalBackend by default; point at a cluster
# to exercise SlurmBackend.
test-integration:
	$(CONDA_ENV_RUN) pytest -v -m "integration" $(PACKAGE_DIR)/tests/

docs:
	$(CONDA_ENV_RUN) mkdocs build

docs-deploy:
ifndef VERSION
	$(error VERSION is not set)
endif
	$(CONDA_ENV_RUN) mike deploy --push --update-aliases $(VERSION)
