import react from "@vitejs/plugin-react";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

// The built page may only load its own files and call its own API (the Control Finder API that
// serves it): no other servers, no inline scripts, no plugins, and no form posts.
export const CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  "connect-src 'self'",
  "img-src 'self' data:",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
].join("; ");

// The API's routes, which the development server passes on to the API on port 8000.
export const API_ROUTES = ["/search", "/answer", "/controls", "/health"];

// Added to the built page only: the development server needs inline scripts for live reloading.
function contentSecurityPolicy(): Plugin {
  return {
    name: "content-security-policy",
    apply: "build",
    transformIndexHtml: (html) => {
      const charset = '<meta charset="UTF-8" />';
      if (!html.includes(charset)) throw new Error("index.html must declare its charset");
      return html.replace(charset, `${charset}\n    <meta http-equiv="Content-Security-Policy" content="${CONTENT_SECURITY_POLICY}" />`);
    },
  };
}

export default defineConfig({
  plugins: [react(), contentSecurityPolicy()],
  server: { proxy: Object.fromEntries(API_ROUTES.map((route) => [route, "http://127.0.0.1:8000"])) },
  test: { environment: "node" },
});
