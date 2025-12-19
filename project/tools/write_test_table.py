from pathlib import Path
p = Path('project/report/test_table.tex')
content = '''\\documentclass{article}
\\usepackage{booktabs}
\\begin{document}
\\begin{tabular}{l r}
\\toprule
Metric & Value \\
\\midrule
Silhouette & 0.57 \\
\\bottomrule
\\end{tabular}
\\end{document}
'''
p.write_text(content)
print('Wrote', p)
