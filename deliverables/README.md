# Deliverables

| Folder | What it is | Format |
|---|---|---|
| `CV_Report/` | Computer Vision report, MAAI 901 Deep Learning for Visual Data (12 sections, 2,500 to 3,000 words) | LaTeX `article` |
| `NLP_Report/` | NLP report, MAAI 902 NLP and LLMs, section 5.3.3 (8 sections, 3,000 to 4,000 words) | LaTeX `article` |
| `IEEE_Paper/` | Unified group research paper for both subjects | LaTeX `IEEEtran` (conference) |

Each folder has the `.tex` source, a `figures/` folder and a PDF.

## Overleaf

1. In Overleaf choose **New Project**, then **Upload Project**, and upload a zip of one document folder (the `.tex` file plus `figures/`).
2. Compile with pdfLaTeX (the default). `IEEEtran.cls` is built into Overleaf.
3. Fill in the group-member placeholders (`[Group member 2]`, `[Group member 3]`) in the author line and in the contributions section.

The PDFs in this repository were rendered from the same `.tex` sources by `build.py` (an HTML conversion printed with Chrome), because no TeX installation was available while drafting. Overleaf's own compile is the official version; line breaks and page counts can differ slightly.

## Scripts

* `make_figures.py` draws the summary figures from `results/tables/*.csv` and copies the notebook's own figures from `results/figures/`.
* `build.py` checks that every citation has a reference (and that no reference is unused), orders the bibliography by first citation, counts body words (tables and references excluded) and renders the PDFs.

Every number in the documents comes from `results/tables/`, the output of the executed notebook.
