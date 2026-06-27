import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { Index } from "./routes";
import "./styles.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element not found");
}

createRoot(root).render(
  <StrictMode>
    <Index />
  </StrictMode>,
);
