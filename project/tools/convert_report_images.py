from PIL import Image
import os
import sys
report_dir = os.path.join(os.getcwd(), 'project', 'report')
pngs = [p for p in os.listdir(report_dir) if p.lower().endswith('.png')]
if not pngs:
    print('No PNGs found in', report_dir); sys.exit(0)
for p in pngs:
    path = os.path.join(report_dir, p)
    im = Image.open(path)
    print('Processing', p, 'mode=', im.mode, 'size=', im.size)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        rgb = Image.new('RGB', im.size, (255,255,255))
        rgb.paste(im, mask=im.split()[-1])
        rgb.save(path)
        print('Converted', p, 'to RGB (alpha removed)')
    elif im.mode != 'RGB':
        im.convert('RGB').save(path)
        print('Converted', p, 'to RGB (mode changed)')
    else:
        print(p, 'already RGB')
print('Done')
