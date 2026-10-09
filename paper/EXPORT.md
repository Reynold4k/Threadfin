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

The native benchmark and external public-data reanalyses are complete. The
current export requires six main figures and sixteen supplementary figures
(22 verified images), including the Figure 2 biological audit. Source Figure 4B
is the LARRY author cell layout, while 4C is a new Threadfin clone-day layout.
Figure 2B retains the distinct historical V–D–J grouping and its stated limits.

~~~bash
python paper/export_manuscript.py --template /path/to/author-template.docx
~~~

Outputs are the DOCX, a matching `.ris`, an `.audit.json`, and the rendered
`MANUSCRIPT_draft_v2.md`. The audit checks exact template styles, author text,
image count and unresolved reference/result placeholders. Native RBD numbers
are read only from the completed saved source summary, with all eight
comparators and consistent mouse/family coverage. Export never runs a new
biological analysis.

The verified 9 October 2026 delivery is saved under paper/release:
Threadfin_MANUSCRIPT_GC_2026-10-09.docx, its matching RIS and its original
export audit. This copy has the same bytes as the workspace export named in
the audit. All 22 embedded images match the current figure-source hashes,
and 29 reference records are resolved. Figure 4E includes six methods/controls,
including RNA mean plus variance.

OOXML validation does not verify Word pagination. Review the rendered pages
before submission, and visually review any regenerated source figures. The current HPC environment has no installed Word or
LibreOffice renderer; the document's audit records that remaining check.
