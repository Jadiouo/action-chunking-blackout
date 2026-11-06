#!/bin/bash
# Build paper/arxiv/arxiv_submission.tar.gz (only files needed to compile) and check it.
# Usage: bash code/make_arxiv_package.sh [--draft]
#   --draft  placeholders (<REPO-URL>, <COMMIT>) only warn; without it they make the script exit 1.
#            Without --draft the script also requires that \repourl is a github.com URL and \repocommit is set (the public repo URL and tag).
# Does not upload anything. Does not modify main.tex. Staging dir: .scratch/arxiv_pkg/
set -u
DRAFT=0; [ "${1:-}" = "--draft" ] && DRAFT=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/paper/arxiv"
WORK="$ROOT/.scratch/arxiv_pkg"
STAGE="$WORK/stage"; TEST="$WORK/test"
OUT="$SRC/arxiv_submission.tar.gz"
FAIL=0; WARN=0
fail() { echo "FAIL: $*"; FAIL=1; }
warn() { echo "WARN: $*"; WARN=1; }

rm -rf "$WORK"; mkdir -p "$STAGE" "$TEST"
TEX="$SRC/main.tex"
[ -f "$TEX" ] || { echo "no $TEX"; exit 2; }

# 1. placeholders
if grep -nE '<REPO-URL>|<COMMIT>' "$TEX" >/dev/null; then
  echo "########################################################################"
  echo "#  WARNING: main.tex still has <REPO-URL> / <COMMIT> placeholders:"
  grep -nE '<REPO-URL>|<COMMIT>' "$TEX" | sed 's/^/#    /'
  echo "#  Fill \\repourl and \\repocommit before a real submission."
  echo "########################################################################"
  if [ $DRAFT -eq 0 ]; then FAIL=1; echo "(exit 1: not --draft)"; fi
fi

if ! grep -qE '^\\newcommand\{\\repourl\}\{\\url\{https://github\.com/[^}]+\}\}' "$TEX" || ! grep -qE '^\\newcommand\{\\repocommit\}\{\\texttt\{[^<>}]+\}\}' "$TEX"; then
  echo "\\repourl / \\repocommit are not set to a github.com URL and a tag"
  if [ $DRAFT -eq 0 ]; then FAIL=1; fi
fi

# 2. collect files that main.tex actually references
cp "$TEX" "$STAGE/main.tex"
if [ -f "$SRC/main.bbl" ]; then
  cp "$SRC/main.bbl" "$STAGE/main.bbl"
  [ "$SRC/main.bbl" -ot "$SRC/refs.bib" ] && warn "main.bbl is older than refs.bib; rebuild first (the build script is not run here)"
else fail "main.bbl missing"; fi
# \input{x} (adds .tex) and \includegraphics[..]{x} (adds .pdf/.png if no extension)
# Comments are ignored when scanning.
SCAN=$(sed -E 's/(^|[^\\])%.*$/\1/' "$TEX")
echo "$SCAN" | grep -oE '\\input\{[^}]+\}' | sed -E 's/\\input\{(.*)\}/\1/' | while read -r f; do
  case "$f" in *.tex) p="$f";; *) p="$f.tex";; esac
  if [ -f "$SRC/$p" ]; then mkdir -p "$STAGE/$(dirname "$p")"; cp "$SRC/$p" "$STAGE/$p"; else echo "MISSING $p"; fi
done > "$WORK/missing.txt"
echo "$SCAN" | grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' | sed -E 's/.*\{([^}]+)\}$/\1/' | while read -r f; do
  p=""
  for c in "$f" "$f.pdf" "$f.png" "$f.jpg"; do [ -f "$SRC/$c" ] && p="$c" && break; done
  if [ -n "$p" ]; then mkdir -p "$STAGE/$(dirname "$p")"; cp "$SRC/$p" "$STAGE/$p"; else echo "MISSING $f"; fi
done >> "$WORK/missing.txt"
if [ -s "$WORK/missing.txt" ]; then cat "$WORK/missing.txt"; fail "referenced files missing"; fi
# nested \input inside the tables (one more level)
for t in $(find "$STAGE" -name '*.tex' ! -name main.tex); do
  grep -oE '\\(input|includegraphics)(\[[^]]*\])?\{[^}]+\}' "$t" >/dev/null && warn "nested include in $t; check by hand"
done

# 3. content checks on the staged files
echo "== content checks"
if grep -rnE '/home/|/Users/|C:\\\\' "$STAGE" --include='*.tex' --include='*.bbl'; then fail "local path in package"; fi
if grep -rnE 'TODO|FIXME|XXX|NEEDS CLARIFICATION' "$STAGE" --include='*.tex' --include='*.bbl'; then fail "TODO-like marker in package"; fi
# non-ASCII inside comments (private Chinese notes etc.)
python3 - "$STAGE" <<'PY' || FAIL=1
import sys,re,glob,os
bad=0
for f in glob.glob(os.path.join(sys.argv[1],'**','*.tex'),recursive=True)+glob.glob(os.path.join(sys.argv[1],'*.bbl')):
    for i,l in enumerate(open(f,encoding='utf8'),1):
        m=re.search(r'(?<!\\)%(.*)$',l)
        if m and re.search(r'[^\x00-\x7f]',m.group(1)):
            print(f"FAIL: non-ASCII in comment {os.path.basename(f)}:{i}: {l.strip()[:80]}"); bad=1
sys.exit(bad)
PY
ncom=$(grep -rcE '(^|[^\\])%' "$STAGE" --include='*.tex' | awk -F: '{s+=$2} END{print s+0}')
echo "comment lines in package: $ncom (review with: grep -rn '%' $STAGE)"
# images other than figures that are not referenced? list package
echo "== package files"; (cd "$STAGE" && find . -type f | sort)

# 4. clean-room compile twice
echo "== compile in fresh dir"
cp -r "$STAGE/." "$TEST/"
(
  cd "$TEST" || exit 1
  # pdflatex needs no network; use a private network namespace when the system allows it.
  RUN="pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex"
  if unshare -rn true 2>/dev/null; then PFX="unshare -rn"; else PFX=""; echo "note: unshare -n not permitted; pdflatex does not use the network anyway"; fi
  HOME="$TEST" TEXMFVAR="$TEST/.texvar" $PFX $RUN >compile1.log 2>&1 && HOME="$TEST" TEXMFVAR="$TEST/.texvar" $PFX $RUN >compile2.log 2>&1
) || { fail "pdflatex failed"; tail -25 "$TEST/compile1.log" "$TEST/compile2.log" 2>/dev/null | grep -E '^!|^l\.|Error' | head; }
if [ -f "$TEST/main.pdf" ] && [ -s "$TEST/compile2.log" ]; then
  P_NEW=$(pdfinfo "$TEST/main.pdf" | awk '/^Pages/{print $2}')
  P_OLD=$(pdfinfo "$SRC/main.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
  echo "pages: package=$P_NEW  main.pdf=$P_OLD"
  [ "$P_NEW" = "$P_OLD" ] || fail "page count differs from main.pdf (main.pdf may be stale)"
  UND=$(grep -cE '(Citation|Reference) .*undefined|There were undefined' "$TEST/compile2.log")
  echo "undefined refs/citations after pass 2: $UND"; [ "$UND" = "0" ] || fail "undefined references"
  # PDF metadata
  pdfinfo "$TEST/main.pdf" | grep -E '^(Title|Author|Subject|Keywords|Creator|Producer)'
  if pdfinfo "$TEST/main.pdf" | grep -E '/home/|/Users/'; then fail "path in PDF metadata"; fi
  if strings "$TEST/main.pdf" | grep -aE '/home/|/Users/'; then fail "path string inside PDF"; fi
else
  fail "no PDF produced"
fi

# 5. tarball
if [ $FAIL -eq 0 ] || [ $DRAFT -eq 1 ]; then
  (cd "$STAGE" && tar --sort=name --mtime='2000-01-01' --owner=0 --group=0 --numeric-owner -czf "$OUT" .)
  echo "== wrote $OUT ($(du -h "$OUT" | cut -f1))"
  tar tzf "$OUT" | sed 's/^/   /'
fi
echo "== result: FAIL=$FAIL WARN=$WARN DRAFT=$DRAFT"
[ $FAIL -eq 0 ] || exit 1
echo "Reminder: preview the PDF on arXiv (Chinese name, all pages) before pressing Submit. Nothing was uploaded."
