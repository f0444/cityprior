#!/bin/sh
# Build paper/main.pdf (needs pdflatex + bibtex, e.g. MacTeX / TeX Live basic).
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
bibtex main >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
echo "built $(pwd)/main.pdf"
