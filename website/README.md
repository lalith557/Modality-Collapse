# Forward or Gradient? — interactive research website

An interactive presentation of the study *"Forward or Gradient? Disentangling
Training-Time Feature-Scale Effects on Modality Contribution."* It walks a
visitor through the scientific story:

> feature scaling → forward representation magnitude → gradient / optimizer
> pathway → parameter updates → modality contribution → multimodal performance

Built as a scientific-instrument-style reading experience, not a dashboard.

## Stack
- React 18 + TypeScript + Vite
- Tailwind CSS (custom scientific design tokens)
- Recharts for restrained, paper-style charts
- Lucide icons

## Run locally
```bash
cd website
npm install
npm run dev
```
Then open the printed local URL.

## Build
```bash
npm run build      # type-checks and outputs static files to dist/
npm run preview    # serve the production build locally
```
The build is fully static and can be hosted on GitHub Pages, Netlify, Vercel, or
any static host. `vite.config.ts` uses a relative `base` so it also works from a
project subpath. Routing uses `HashRouter`, so deep links work without server
rewrites.

## Scientific integrity
Every experimental value shown comes from the manuscript (Tables I–III and text)
and its `results/**` artifacts, and lives in a single source of truth:
[`src/data/research.ts`](src/data/research.ts). Nothing is invented, and no live
training is run. The Experiment Lab is an interactive visualization: its result
endpoints are the reported values for the selected configuration, and the α
slider / training curve are labelled schematic.

## Structure
```
src/
  components/   Sidebar, Layout, primitives, diagrams, controls, figures
  charts/       Recharts figures + shared theme
  pages/        the ten routes (Home, Overview, Experiment Lab, …)
  data/         research.ts (verified values) + nav.ts
  lib/          number formatting
public/
  main.pdf      the compiled manuscript
```
