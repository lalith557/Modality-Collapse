# Website

A single-page React application that presents the study and hosts the Prediction Studio.

| Route | Page |
|---|---|
| `/` | Home |
| `/overview` | Research overview and key findings |
| `/lab` | Experiment Lab: every measured configuration, with the logged per-epoch validation curves |
| `/studio` | Prediction Studio: runs the saved checkpoints on uploaded files (needs the backend, see [`../docs/PREDICTION_STUDIO.md`](../docs/PREDICTION_STUDIO.md)) |
| `/architecture` | The two real-data models layer by layer, in a 2D view and an interactive 360° three.js scene |
| `/disentanglement` | Forward-versus-gradient decomposition, optimizer and fusion analyses |
| `/real-world` | CREMA-D and UTD-MHAD evidence with decoupled metrics |
| `/methodology` | Protocol, metrics and statistics |
| `/paper` | Section summaries and the PDF of the CA manuscript |

## Stack

React 18, TypeScript, Vite 5, Tailwind CSS, Recharts, three.js (loaded as a separate chunk on
the architecture page), Vitest and Testing Library. Routing uses `HashRouter`, and
`vite.config.ts` sets a relative `base`, so the build runs from any static host or a GitHub
Pages project path without rewrites.

## Development

Node.js 18 or newer.

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # Vitest
npm run build      # type-check + production build into dist/
npm run preview    # serve dist/ on http://localhost:4173
```

The Prediction Studio calls `http://127.0.0.1:8000` by default; set `VITE_STUDIO_API` to use
another backend address. Every other page is static and works without the backend.

## Where the numbers come from

No research value on the site is typed into a component. They live in a few data files, each
checked against the frozen results:

| File | Produced by | Checked by |
|---|---|---|
| `src/data/research.ts` | transcribed from `results/` and `docs/CLAIMS_LEDGER.md` | `src/data/evidence.test.ts` against `evidence.verified.json` |
| `src/data/evidence.verified.json` | `python scripts/export_site_evidence.py` | `tests/test_site_artifacts.py` (`--check`) |
| `src/data/architecture.generated.json` | `python scripts/export_architecture.py` (read from the PyTorch modules and checkpoints) | `tests/test_site_artifacts.py` (`--check`) |
| `src/data/trajectories.generated.json` | `python scripts/export_lab_trajectories.py` (per-epoch logs) | `tests/test_site_artifacts.py` (`--check`) |
| `public/arch/*.png`, `src/data/architecture.media.json` | `python scripts/export_architecture_media.py` (real preprocessed inputs) | |

The Python scripts run from the repository root.

## Layout

```
src/
├── pages/          one component per route
├── architecture/   2D diagram, three.js scene, model and stage metadata
├── studio/         Prediction Studio client: API calls, upload validation, result views
├── charts/         Recharts figures and shared theme
├── components/     layout, navigation, shared UI, disentanglement sections
├── data/           research.ts, nav.ts and the generated JSON files above
├── context/, state/, lib/
└── main.tsx, App.tsx
public/
├── ca_research_paper.pdf
└── arch/           input thumbnails for the architecture page
```
