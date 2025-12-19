Local LaTeX compilation

If you want to compile `report/report.tex` to PDF locally using a TeX engine (recommended for final submission):

1. Install a TeX distribution (e.g., TeX Live or MiKTeX).
2. In the `project/report` folder, run:

   pdflatex -interaction=nonstopmode -output-directory out report.tex
   pdflatex -interaction=nonstopmode -output-directory out report.tex

3. The compiled PDF will be available at `project/report/out/report.pdf`.

Notes:
- `pdflatex` is not available in the current remote environment, so an HTML→PDF conversion was used instead (`results/final_report_submission.pdf`).
- If you permit installation of a TeX distribution here, I can attempt to install and compile it remotely (this requires significant disk space and network access).