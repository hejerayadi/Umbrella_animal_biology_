// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - TanStack devtools (dev-only, first), tanstackStart, viteReact, tailwindcss, tsConfigPaths,
//     nitro (build-only using cloudflare as a default target), VITE_* env injection, @ path alias,
//     React/TanStack dedupe, error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... }, etc... }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";

const DEV_SERVER_PORT = 5173;

const withLovableDefaults = defineConfig({
  vite: {
    optimizeDeps: {
      // The MFA route is lazy-loaded. Pre-bundle its dependencies so a Vite
      // optimizer restart cannot leave the route pointing at stale chunks.
      include: ["input-otp", "qrcode.react"],
    },
    resolve: {
      tsconfigPaths: true,
    },
    server: {
      host: "127.0.0.1",
      port: DEV_SERVER_PORT,
      strictPort: true,
    },
  },
  tanstackStart: {
    // Redirect TanStack Start's bundled server entry to src/server.ts (our SSR error wrapper).
    // nitro/vite builds from this
    server: { entry: "server" },
  },
});

export default async (...args: Parameters<typeof withLovableDefaults>) => {
  const config = await withLovableDefaults(...args);

  // Vite 8 resolves tsconfig paths natively. The Lovable wrapper still adds
  // the legacy plugin, so remove it to avoid duplicate resolution and warnings.
  config.plugins = config.plugins?.filter(
    (plugin) =>
      !(
        plugin &&
        typeof plugin === "object" &&
        "name" in plugin &&
        plugin.name === "vite-tsconfig-paths"
      ),
  );

  return config;
};
