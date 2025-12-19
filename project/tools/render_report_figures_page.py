from PIL import Image, ImageDraw, ImageFont
import os
# choose results directory under workspace root (cwd is usually workspace root)
out_dir = os.path.join(os.getcwd(), 'results')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)
# locate the report images directory by searching for 'report.tex' under the workspace
report_dir = None
for dirpath, dirnames, filenames in os.walk(os.getcwd()):
    if 'report.tex' in filenames and os.path.basename(dirpath).lower() == 'report':
        report_dir = dirpath
        break
# fallback: find any folder named 'report' containing images
if report_dir is None:
    for dirpath, dirnames, filenames in os.walk(os.getcwd()):
        if os.path.basename(dirpath).lower() == 'report':
            report_dir = dirpath
            break
if report_dir is None:
    raise SystemExit('Could not find report directory via search under ' + os.getcwd())
# helper to pick available extension
def pick(name):
    for ext in ['.png', '.jpg', '.pdf']:
        p = os.path.join(report_dir, name+ext)
        if os.path.exists(p):
            return p
    raise SystemExit('Missing image for ' + name)
# Files to use
sil = pick('silhouette_summary')
umap1 = pick('umap_ld64')
umap2 = pick('umap_ld16')
tsne1 = pick('tsne_ld64')
tsne2 = pick('tsne_ld16')
imgs = [sil, umap1, umap2, tsne1, tsne2]
for p in imgs:
    if not os.path.exists(p):
        print('Missing', p)
        raise SystemExit(1)
# Create canvas
W, H = 1200, 1600
bg = Image.new('RGB', (W, H), 'white')
d = ImageDraw.Draw(bg)
# Title
try:
    font = ImageFont.truetype('arial.ttf', 36)
except Exception:
    font = ImageFont.load_default()
d.text((40, 20), 'Figures', fill='black', font=font)
# Paste silhouette at top
sil_im = Image.open(sil).convert('RGB')
max_w = W - 80
scale = min(max_w / sil_im.width, 400 / sil_im.height)
nw = int(sil_im.width * scale)
nh = int(sil_im.height * scale)
sil_im = sil_im.resize((nw, nh), Image.LANCZOS)
bg.paste(sil_im, ((W-nw)//2, 80))
# Below: two rows with two images each
y = 80 + nh + 40
pair_w = (W - 3*40) // 2
pair_h = 420
def place(imgpath, x, y, dw, dh):
    im = Image.open(imgpath).convert('RGB')
    scale = min(dw / im.width, dh / im.height)
    new = im.resize((int(im.width*scale), int(im.height*scale)), Image.LANCZOS)
    bg.paste(new, (x + (dw-new.width)//2, y + (dh-new.height)//2))
# row1: umap1, umap2
place(umap1, 40, y, pair_w, pair_h)
place(umap2, 40 + pair_w + 40, y, pair_w, pair_h)
# row2: tsne1, tsne2
y2 = y + pair_h + 30
place(tsne1, 40, y2, pair_w, pair_h)
place(tsne2, 40 + pair_w + 40, y2, pair_w, pair_h)
# Captions
smallf = ImageFont.load_default()
d.text((40, y + pair_h + 5), 'UMAP: left = conv_ld64_hc32, right = conv_ld16_hc16', fill='black', font=smallf)
d.text((40, y2 + pair_h + 5), 't-SNE: left = conv_ld64_hc32, right = conv_ld16_hc16', fill='black', font=smallf)
# Save
out = os.path.join(out_dir, 'report_figures_page_render.png')
bg.save(out)
print('Wrote', out)
