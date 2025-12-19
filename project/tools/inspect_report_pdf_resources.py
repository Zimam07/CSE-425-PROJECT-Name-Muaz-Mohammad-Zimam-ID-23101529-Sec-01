b = open('results/report.pdf','rb').read()
for i in range(len(b)):
    if b.startswith(b'/XObject', i):
        s=b[max(0,i-150):i+300]
        print(s.replace(b"\n", b"\\n")[:500])
        print('-'*80)
        break
