from PyPDF2 import PdfReader
p='results/report.pdf'
reader=PdfReader(p)
print('pages', len(reader.pages))
for i,page in enumerate(reader.pages):
    resources = page.get('/Resources') or {}
    xobj = resources.get('/XObject') if resources else None
    print('page', i, 'has XObject?', bool(xobj))
    if xobj:
        print('XObject keys', list(xobj.keys()))
    # also print content length
    cont = page.get_contents()
    print('content obj', type(cont))
    try:
        text = page.extract_text()
        print('page text length', len(text) if text else 0)
    except Exception as e:
        print('extract_text error', e)
