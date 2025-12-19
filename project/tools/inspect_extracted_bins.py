from pathlib import Path
for p in sorted(Path('results').glob('extracted_image_*.bin'))[:6]:
    b = p.read_bytes()
    sig = ''.join(f'{x:02X}' for x in b[:8])
    head = ''.join(chr(x) if 32 <= x <= 126 else '.' for x in b[:24])
    print(p.name, 'len=', len(b), 'sig=', sig, 'head=', head)
