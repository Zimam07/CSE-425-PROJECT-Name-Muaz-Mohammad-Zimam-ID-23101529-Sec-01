from pathlib import Path
b = Path('project/report/report.tex').read_bytes()
idx = b.find(b'\\n')
print('first index of backslash-n:', idx)
# show a slice around it
print(b[idx-20:idx+40])
print(list(b[idx-10:idx+10]))
