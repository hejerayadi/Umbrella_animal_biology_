import { Link } from "@tanstack/react-router";
import type { ReactNode } from "react";

import { UmbrellaLogo } from "./logo";
import { ThemeToggle } from "./theme-toggle";

export function AuthShell({ children, width = "max-w-sm" }: { children: ReactNode; width?: string }) {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="flex items-center justify-between px-5 py-5">
        <Link to="/"><UmbrellaLogo /></Link><ThemeToggle />
      </header>
      <main className="flex flex-1 items-center justify-center px-5 pb-16">
        <div className={`w-full ${width}`}>{children}</div>
      </main>
    </div>
  );
}

