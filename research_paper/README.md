# Papers

Two IEEE conference manuscripts (IEEEtran, `\documentclass[conference]`) describe this work.

| Source | Title | Pages | Covers |
|---|---|---|---|
| `main.tex` / `main.pdf` | *Feature Scale as a Hidden Per-Modality Learning Rate: When Masked-Accuracy Dominance Misleads* | 6 | Phases 1 to 4: the operator decomposition, Proposition 1, synthetic mechanism tests, CREMA-D with decoupled metrics, positive control |
| `ca_paper.tex` | *When Scale Silences a Modality: Disentangling Representation Magnitude, Optimization Dynamics, and Evaluation Artifacts in Multimodal Collapse* | 7 | Phases 0 to 3, early-ratio diagnostics, the Adam/SGD contrast, the zero-masking artifact on CREMA-D |

The website's Paper page serves a compiled copy of the second manuscript from
`../website/public/ca_research_paper.pdf`.

```
research_paper/
├── main.tex, main.pdf     # 6-page manuscript; bibliography embedded (thebibliography)
├── ca_paper.tex           # 7-page manuscript; plots are pgfplots with values taken from results/
├── references.bib         # BibTeX source of main.tex's embedded bibliography
├── IEEEtran.cls           # IEEE conference class, v062824
├── figures/
│   ├── architecture_diagram.jpg                     # ca_paper Fig. 1 (conceptual overview)
│   ├── fig_synth.pdf, fig_real.pdf                  # main.tex Figs. 2 and 3
│   └── fig_disentangle.pdf, fig_fusion.pdf, fig_utd.pdf
└── scripts/
    ├── make_paper_figs.py # fig_synth / fig_real from results/phase2_mechanism and results/phase3_cremad
    └── make_figs.py       # fig_disentangle / fig_fusion / fig_utd from the decomposition and UTD-MHAD runs
```

## Build

```bash
cd research_paper
pdflatex main && pdflatex main
pdflatex ca_paper && pdflatex ca_paper
```

`main.tex` embeds its bibliography, so BibTeX is not needed. To regenerate it from
`references.bib`, replace the `thebibliography` block with `\bibliography{references}` and run
`pdflatex`, `bibtex`, `pdflatex`, `pdflatex`. Two entries in `references.bib` are marked
`VERIFY`; check their bibliographic details against the originals before submission.

## Figures

The figures contain no hand-entered values. From the repository root:

```bash
python research_paper/scripts/make_paper_figs.py
python research_paper/scripts/make_figs.py
```

The plots inside `ca_paper.tex` are drawn with pgfplots; comments in the source name the
result files behind the plotted values. Every number in both manuscripts traces to
[`../docs/CLAIMS_LEDGER.md`](../docs/CLAIMS_LEDGER.md) and the commands in
[`../docs/REPRODUCING.md`](../docs/REPRODUCING.md).
