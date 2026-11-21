import {
  Home,
  BookOpen,
  FlaskConical,
  GitFork,
  Database,
  ClipboardList,
  FileText,
  ScanSearch,
  Network,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  path: string;
  label: string;
  icon: LucideIcon;
  group: "overview" | "experiments" | "models" | "evidence" | "reference";
}

/** Primary navigation, in display order. Optimizer Analysis and Fusion & Utility are
 * sections of Disentanglement; Findings is a section of Research Overview. */
export const nav: NavItem[] = [
  { path: "/", label: "Home", icon: Home, group: "overview" },
  { path: "/overview", label: "Research Overview", icon: BookOpen, group: "overview" },
  { path: "/lab", label: "Experiment Lab", icon: FlaskConical, group: "experiments" },
  { path: "/studio", label: "Prediction Studio", icon: ScanSearch, group: "models" },
  { path: "/architecture", label: "Model Architecture", icon: Network, group: "models" },
  { path: "/disentanglement", label: "Disentanglement", icon: GitFork, group: "evidence" },
  { path: "/real-world", label: "Real-World Evidence", icon: Database, group: "evidence" },
  { path: "/methodology", label: "Methodology", icon: ClipboardList, group: "reference" },
  { path: "/paper", label: "Paper", icon: FileText, group: "reference" },
];

export const groupLabels: Record<NavItem["group"], string> = {
  overview: "Overview",
  experiments: "Experiments",
  models: "Models",
  evidence: "Evidence",
  reference: "Reference",
};
