"""Print the plain-text arXiv abstract from paper/arxiv/main.tex and its length (limit 1920)."""
import re, pathlib
s = pathlib.Path(__file__).resolve().parents[1].joinpath('paper/arxiv/main.tex').read_text()
a = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', s, re.S).group(1).strip()
a = re.sub(r'\$([^$]*)\$', lambda m: re.sub(r'\\mathrm\{(\w+)\}', r'\1', m.group(1)).replace('{', '').replace('}', '').replace('\\', ''), a)
a = re.sub(r'\\textsf\{(.*?)\}', r'\1', a).replace('\\emph', '').replace('--', '-').replace('\\%', '%')
a = re.sub(r'\s+', ' ', a)
print(a); print(f'\n[{len(a)} chars, limit 1920, non-ASCII: {sum(ord(c)>127 for c in a)}]')
