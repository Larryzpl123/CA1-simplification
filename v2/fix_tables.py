"""Pandoc's docx writer emits tables with an EMPTY <w:tblGrid> -- no <w:gridCol>
elements at all. Word and LibreOffice then have to guess the column layout, and
they guess badly: the right-hand columns are pushed off the page and are simply
invisible in the rendered document. The data is in the file. The reader cannot
see it. python-docx reports len(table.columns) == 0 for these, which is the tell.

Fix: build the tblGrid from the actual cells, and set BOTH the grid widths and
every cell width in DXA. docx ignores one without the other."""
import sys
from docx import Document
from docx.shared import Twips
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

USABLE = 12240 - 2 * 1008          # US Letter minus 0.7in margins, in DXA

doc = Document(sys.argv[1])
fixed = 0
for t in doc.tables:
    rows = t._tbl.findall(qn('w:tr'))
    if not rows:
        continue
    ncol = len(rows[0].findall(qn('w:tc')))
    if ncol == 0:
        continue

    # weight each column by its longest cell, clamped so nothing starves
    text = [[ "".join(n.text or "" for n in tc.iter(qn('w:t')))
              for tc in r.findall(qn('w:tc')) ] for r in rows]
    w = []
    for c in range(ncol):
        L = max((len(row[c]) for row in text if len(row) > c), default=4)
        w.append(max(5, min(L, 30)))
    tot = sum(w)
    dxa = [int(USABLE * x / tot) for x in w]
    dxa[-1] += USABLE - sum(dxa)

    tblPr = t._tbl.tblPr
    for tag in ('w:tblW', 'w:tblLayout'):
        old = tblPr.find(qn(tag))
        if old is not None:
            tblPr.remove(old)
    tw = OxmlElement('w:tblW'); tw.set(qn('w:w'), str(USABLE)); tw.set(qn('w:type'), 'dxa')
    tl = OxmlElement('w:tblLayout'); tl.set(qn('w:type'), 'fixed')
    tblPr.append(tw); tblPr.append(tl)

    # REBUILD the grid -- pandoc left it empty
    grid = t._tbl.find(qn('w:tblGrid'))
    if grid is None:
        grid = OxmlElement('w:tblGrid')
        t._tbl.insert(list(t._tbl).index(tblPr) + 1, grid)
    for gc in list(grid):
        grid.remove(gc)
    for width in dxa:
        gc = OxmlElement('w:gridCol'); gc.set(qn('w:w'), str(width)); grid.append(gc)

    for r in rows:
        for tc, width in zip(r.findall(qn('w:tc')), dxa):
            tcPr = tc.find(qn('w:tcPr'))
            if tcPr is None:
                tcPr = OxmlElement('w:tcPr'); tc.insert(0, tcPr)
            old = tcPr.find(qn('w:tcW'))
            if old is not None:
                tcPr.remove(old)
            tcw = OxmlElement('w:tcW')
            tcw.set(qn('w:w'), str(width)); tcw.set(qn('w:type'), 'dxa')
            tcPr.insert(0, tcw)
    fixed += 1

doc.save(sys.argv[1])
print(f"rebuilt tblGrid + cell widths on {fixed} tables ({USABLE/1440:.2f} in usable)")
