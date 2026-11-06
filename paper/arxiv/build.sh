#!/bin/bash
# Build paper/arxiv/main.pdf (arXiv-style single-column draft).
# Uses latexmk (pdflatex + bibtex). Run: bash paper/arxiv/build.sh
set -e
cd "$(dirname "$0")"
latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex > build_main.log 2>&1 || { tail -40 build_main.log; exit 1; }
echo "== main: pages $(pdfinfo main.pdf | awk '/Pages/{print $2}')"
echo "undefined citations/references: $(grep -cE '(Citation|Reference) .* undefined' main.log || true)"
grep -E "Warning--|Overfull" main.log main.blg | head -20 || true
