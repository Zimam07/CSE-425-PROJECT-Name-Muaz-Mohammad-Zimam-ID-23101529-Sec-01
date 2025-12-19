from pathlib import Path
s = Path('project/report/report.tex').read_text()
lines = s.splitlines()
for i in range(36, 44):
    print(i+1, repr(lines[i]))
