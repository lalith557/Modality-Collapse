import { HashRouter, Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { Overview } from "./pages/Overview";
import { ExperimentLab } from "./pages/ExperimentLab";
import { Disentanglement } from "./pages/Disentanglement";
import { OptimizerAnalysis } from "./pages/OptimizerAnalysis";
import { RealWorld } from "./pages/RealWorld";
import { FusionUtility } from "./pages/FusionUtility";
import { Findings } from "./pages/Findings";
import { Methodology } from "./pages/Methodology";
import { Paper } from "./pages/Paper";

export function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/lab" element={<ExperimentLab />} />
          <Route path="/disentanglement" element={<Disentanglement />} />
          <Route path="/optimizer" element={<OptimizerAnalysis />} />
          <Route path="/real-world" element={<RealWorld />} />
          <Route path="/fusion" element={<FusionUtility />} />
          <Route path="/findings" element={<Findings />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/paper" element={<Paper />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}
