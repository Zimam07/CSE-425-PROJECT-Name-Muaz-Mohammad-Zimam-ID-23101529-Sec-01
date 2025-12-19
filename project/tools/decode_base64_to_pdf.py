import base64
from pathlib import Path
src = Path('../../results/report_base64.txt')
out = Path('../../results/report_from_b64.pdf')
if not src.exists():
    print('Base64 file not found:', src)
else:
    data = src.read_bytes()
    # allow if newline-wrapped
    data = b''.join(data.split())
    pdf = base64.b64decode(data)
    out.write_bytes(pdf)
    print('Wrote', out)
