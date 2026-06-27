import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const base = process.env.GITHUB_PAGES === "true" ? "/APCyc/" : "/";

export default defineConfig({
  base,
  resolve: {
    tsconfigPaths: true,
  },
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
