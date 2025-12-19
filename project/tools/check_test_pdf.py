b = open('results/test_img.pdf','rb').read()
print('len',len(b),'Subtype/Image', b.count(b'/Subtype /Image'), '/XObject', b.count(b'/XObject'))
