import asyncio
from playwright.async_api import async_playwright
import sys

async def run(input_html, output_pdf):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto('file://' + input_html)
        await page.pdf(path=output_pdf, format='A4')
        await browser.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python scripts/html_to_pdf.py input.html output.pdf')
        sys.exit(1)
    in_html = sys.argv[1]
    out_pdf = sys.argv[2]
    asyncio.run(run(in_html, out_pdf))
