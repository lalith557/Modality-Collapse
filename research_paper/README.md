# Forward or Gradient? — IEEE MLSP paper (Overleaf project)

Overleaf-ready LaTeX source for the paper *"Forward or Gradient? A Controlled
Causal Study of Training-Time Feature-Scale Effects on Modality Contribution."*

## Contents
```
research_paper/
├── main.tex            # IEEEtran conference paper (6 pages)
├── references.bib      # 45 references (BibTeX)
├── BUILD_REPORT.md     # compile + audit results (page count, refs, citation order)
├── main.pdf            # compiled 6-page PDF (regenerable)
├── MANUSCRIPT_REVISION_AUDIT.md   # pre-revision audit vs experiment artifacts
├── FINAL_REVIEW_REPORT.md         # post-revision review + verification checklist
├── figures/            # vector figures used by the paper
│   ├── fig_disentangle.pdf   (Fig. 2, full width)
│   ├── fig_fusion.pdf        (Fig. 3)
│   └── fig_utd.pdf           (Fig. 4)
└── scripts/
    └── make_figs.py    # regenerates figures/ from ../results/**/*.json
```
Figure 1 is drawn inline with TikZ (no external file). `make_figs.py` can also emit
`fig_phase.pdf` and `fig_cremad.pdf` (data-sourced), which are not included in the
6-page version; their results are reported in the text and Tables II–III.

## Build
On Overleaf: upload the folder, set the compiler to **pdfLaTeX**, main document
`main.tex`, and compile (Overleaf runs BibTeX automatically).

Locally:
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
# or: latexmk -pdf main.tex
```

## Regenerate figures (optional)
Figures are produced only from the machine-readable experiment results and
contain no hand-entered data. From the repository root:
```bash
python research_paper/scripts/make_figs.py
```
This reads `results/**/*.json` (e.g. `results/cremad_disentangle/…`,
`results/sgd_disentangle/…`, `results/fusion_robustness/…`,
`results/utd_mhad/…`, `results/e4/…`) and writes `research_paper/figures/*.pdf`.

## Notes
- Author name/affiliation and citation details are provided as-is; edit the
  `\author{...}` block and `references.bib` for camera-ready submission.
- Every numerical result traces to a committed result artifact; see
  `BUILD_REPORT.md` and the project-level `EXPERIMENT_FREEZE_AUDIT.md`.
