import re
import zlib
from pathlib import Path
p = Path('results/report.pdf')
b = p.read_bytes()
obj_re = re.compile(rb'(\d+)\s+0\s+obj\s*<<([^>]*)>>\s*stream\s*(.*?)\s*endstream', re.S)
for m in obj_re.finditer(b):
    hdr = m.group(2)
    stream = m.group(3)
    if b'/Subtype/Image' in hdr or b'/Filter' in hdr:
        # basic metadata
        objnum = m.group(1).decode('ascii')
        width = re.search(rb'/Width\s+(\d+)', hdr)
        height = re.search(rb'/Height\s+(\d+)', hdr)
        filter_ = re.search(rb'/Filter/([A-Za-z0-9]+)', hdr)
        cs = re.search(rb'/ColorSpace/([A-Za-z0-9]+)', hdr)
        print('obj', objnum, 'WxH', width.group(1) if width else b'?', 'x', height.group(1) if height else b'?', 'filter', filter_.group(1) if filter_ else b'None', 'CS', cs.group(1) if cs else b'None')
        # try to save JPEG streams directly
        if filter_ and filter_.group(1) == b'DCTDecode':
            out = Path(f'results/extracted_image_{objnum}.jpg')
            out.write_bytes(stream)
            print('wrote', out)
        elif filter_ and filter_.group(1) == b'FlateDecode':
            # attempt to decompress and save raw PNG-like data
            try:
                dec = zlib.decompress(stream)
                out = Path(f'results/extracted_image_{objnum}.bin')
                out.write_bytes(dec)
                print('wrote', out, ' (raw decompressed bytes)')
            except Exception as e:
                print('decompress failed for', objnum, e)
print('done')
