import { HashRouter, Navigate, Routes, Route } from "react-router-dom";
import { ResearchProvider } from "./state/research-context";
import { PageContextProvider } from "./state/page-context";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { Overview } from "./pages/Overview";
import { ExperimentLab } from "./pages/ExperimentLab";
import { Disentanglement } from "./pages/Disentanglement";
import { RealWorld } from "./pages/RealWorld";
import { Methodology } from "./pages/Methodology";
import { Paper } from "./pages/Paper";
import { PredictionStudio } from "./pages/PredictionStudio";
import { ModelArchitecture } from "./pages/ModelArchitecture";
import { StudioProvider } from "./studio/StudioContext";

/** Retired pages: their content now lives in a section of another page. */
export const REDIRECTS: Record<string, string> = {
  "/findings": "/overview?section=findings",
  "/optimizer": "/disentanglement?section=adam-vs-sgd",
  "/fusion": "/disentanglement?section=fusion",
};

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/overview" element={<Overview />} />
        <Route path="/lab" element={<ExperimentLab />} />
        <Route path="/studio" element={<PredictionStudio />} />
        <Route path="/architecture" element={<ModelArchitecture />} />
        <Route path="/disentanglement" element={<Disentanglement />} />
        <Route path="/real-world" element={<RealWorld />} />
        <Route path="/methodology" element={<Methodology />} />
        <Route path="/paper" element={<Paper />} />
        {Object.entries(REDIRECTS).map(([from, to]) => (
          <Route key={from} path={from} element={<Navigate to={to} replace />} />
        ))}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

export function App() {
  return (
    <ResearchProvider>
      <StudioProvider>
        <PageContextProvider>
          <HashRouter>
            <AppRoutes />
          </HashRouter>
        </PageContextProvider>
      </StudioProvider>
    </ResearchProvider>
  );
}
