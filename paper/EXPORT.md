# Manuscript DOCX and reference export

`MANUSCRIPT_source.md` is the editable manuscript. Citations use registry keys
from `references.json`; export assigns parenthetical numbers in first-citation
order and writes a matching EndNote-importable RIS bibliography. Preprints are
explicitly marked. The Word citations are static numbers, with no copied or
fabricated EndNote field records.

The supplied author template is
`Progress_Review_Year2_ChenZhu_v5.docx`. Its Normal, heading, FigureLegend and
EndNoteBibliography style definitions are preserved: Times New Roman 12-point
body, 1.5 line spacing, A4 body pages with 2 cm margins, 10-point figure legends
and 10-point references with the template's hanging indent. Each figure is
embedded intact on its own page; the full legend follows on a separate page.
The author is Chen Zhu at the Department of Microbiology and Immunology,
Peter Doherty Institute for Infection and Immunity, University of Melbourne.
Additional authors and declarations await author confirmation.

Install export dependencies in a document environment:

```bash
python -m pip install python-docx Pillow
python paper/export_manuscript.py --template /path/to/author-template.docx --allow-pending
```

The pending export is labelled `REVIEW_PENDING_RBD`. It includes Figures 1–5
and Supplementary Figures 1–6 and 8, plus explicit placeholders for Figure 6/S7.
Figure 2B and Supplementary 8B–D use the audited real-data GSE246382
centroid analysis; Supplementary 8A preserves the historical notebook Top2a
output separately, with unresolved exact cohort/day attribution. The export checks 12 available images
(14 once the native benchmark figures are complete).
Pending native figures are excluded even if stale images exist. Without
`--allow-pending`, export refuses incomplete native results or missing figures:

```bash
python paper/export_manuscript.py --template /path/to/author-template.docx
```

Outputs are the DOCX, a matching `.ris`, an `.audit.json`, and the rendered
`MANUSCRIPT_draft_v2.md`. The audit checks exact template styles, author text,
image count and unresolved reference/result placeholders. Native RBD numbers
are read only from the completed saved source summary, with all eight
comparators and consistent mouse/family coverage. Export never runs a new
biological analysis. On the current HPC workspace, the existing dependent
finalization job also refreshes the DOCX/RIS after its complete tables and all
planned figures are saved. This uses the author's document environment and
template only when both are present; it does not submit another model run.

OOXML validation does not verify Word pagination. Review the rendered pages
before submission, and visually review the new Figure 6/S7 after the native
benchmark finishes. The current HPC environment has no installed Word or
LibreOffice renderer; the document's audit records that remaining check.
