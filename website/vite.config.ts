import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Base path is relative so the built site works under a GitHub Pages
// project subpath as well as at a domain root.
export default defineConfig({
  plugins: [react()],
  base: "./",
});
