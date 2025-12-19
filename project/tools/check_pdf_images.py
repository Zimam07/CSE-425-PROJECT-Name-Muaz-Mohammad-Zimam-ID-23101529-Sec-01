p='results/report.pdf'
with open(p,'rb') as f:
    b=f.read()
print('bytes length', len(b))
print('Subtype /Image occurrences:', b.count(b'/Subtype /Image'))
print("'/Image' occurrences:", b.count(b'/Image'))
print('/XObject occurrences:', b.count(b'/XObject'))
print('/Filter occurrences:', b.count(b'/Filter'))
print('/DCTDecode occurrences:', b.count(b'/DCTDecode'))
print('/FlateDecode occurrences:', b.count(b'/FlateDecode'))
# show snippets around occurrences
for token in (b'/Subtype /Image', b'/Subtype/Image', b'/Image'):
    print('\n--- Context for', token, '---')
    idx = 0
    found = False
    while True:
        i = b.find(token, idx)
        if i == -1:
            break
        start = max(0, i-120)
        end = min(len(b), i+240)
        print(b[start:end].replace(b"\n", b"\\n")[:800])
        found = True
        idx = i+1
    if not found:
        print('none')
