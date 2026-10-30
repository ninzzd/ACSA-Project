#!/bin/bash
# Verilator lint of the full IPU (must be clean).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
verilator --lint-only -Wall -Wno-DECLFILENAME -I"$ROOT/src" $(ls "$ROOT"/src/ipu_*.sv) --top-module ipu_top && echo "lint clean"
