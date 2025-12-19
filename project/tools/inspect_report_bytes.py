p = 'project/report/report.tex'
with open(p,'rb') as f:
    data = f.read()
print(data[300:420])
print(repr(data[300:420]))
# Show lines with indices
s = data.decode('utf-8', errors='replace')
lines = s.splitlines()
for ln in range(30, 52):
    if ln-1 < len(lines):
        print(ln, repr(lines[ln-1]))

