from PIL import Image
import os
files = ['project/report/silhouette_summary.png','project/report/umap_ld64.png','project/report/umap_ld16.png','project/report/tsne_ld64.png','project/report/tsne_ld16.png']
for f in files:
    p = os.path.join(os.getcwd(), f)
    try:
        im = Image.open(p)
        pixels = list(im.getdata())
        unique = len(set(pixels))
        max_alpha = None
        if im.mode == 'RGBA':
            alphas = [px[3] for px in pixels]
            max_alpha = max(alphas)
        print(p, 'size=', im.size, 'mode=', im.mode, 'unique_pixels=', unique, 'max_alpha=', max_alpha)
    except Exception as e:
        print('ERROR', f, e)
