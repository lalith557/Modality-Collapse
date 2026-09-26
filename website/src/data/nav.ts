import {
  Home,
  BookOpen,
  FlaskConical,
  GitFork,
  Sliders,
  Database,
  Combine,
  Lightbulb,
  ClipboardList,
  FileText,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  path: string;
  label: string;
  icon: LucideIcon;
  group: "overview" | "experiments" | "evidence" | "reference";
}

export const nav: NavItem[] = [
  { path: "/", label: "Home", icon: Home, group: "overview" },
  { path: "/overview", label: "Research Overview", icon: BookOpen, group: "overview" },
  { path: "/lab", label: "Experiment Lab", icon: FlaskConical, group: "experiments" },
  { path: "/disentanglement", label: "Disentanglement", icon: GitFork, group: "experiments" },
  { path: "/optimizer", label: "Optimizer Analysis", icon: Sliders, group: "experiments" },
  { path: "/real-world", label: "Real-World Evidence", icon: Database, group: "evidence" },
  { path: "/fusion", label: "Fusion & Utility", icon: Combine, group: "evidence" },
  { path: "/findings", label: "Findings", icon: Lightbulb, group: "reference" },
  { path: "/methodology", label: "Methodology", icon: ClipboardList, group: "reference" },
  { path: "/paper", label: "Paper", icon: FileText, group: "reference" },
];

export const groupLabels: Record<NavItem["group"], string> = {
  overview: "Overview",
  experiments: "Experiments",
  evidence: "Evidence",
  reference: "Reference",
};
