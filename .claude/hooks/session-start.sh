#!/bin/bash
# SessionStart hook for slimcity — prepares a Claude Code on the web container
# so that experiments (qd_train.py / record.py / the ELM harness) run end-to-end.
#
# It installs the SWIG build dependency, the Python packages, and compiles the
# headless Micropolis C++ engine into engine/build/. The wrapper (slimcity.py)
# inserts engine/build on sys.path itself, so no PYTHONPATH is needed.
#
# Idempotent and non-interactive: safe to run on every session start.
set -euo pipefail

# Only run in Claude Code on the web (remote) containers. Local machines that
# already have the engine built and deps installed are left untouched.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# 1. System build dependency: SWIG generates engine/build/micropolisengine_wrap.cpp.
if ! command -v swig >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -qq && sudo apt-get install -y -qq swig
  else
    apt-get update -qq && apt-get install -y -qq swig
  fi
fi

# 2. Python dependencies.
#    numpy + ribs (pyribs) drive the CMA-ME search; pillow is for record.py GIFs;
#    anthropic powers the ELM code-genome harness (needs ANTHROPIC_API_KEY at run time).
python3 -m pip install --quiet numpy ribs pillow anthropic

# 3. Build the headless Micropolis engine -> engine/build/_micropolisengine*.so
make -C engine

# 4. Verify the wrapper imports against the freshly built engine.
python3 -c "import slimcity; print('slimcity engine import OK')"
