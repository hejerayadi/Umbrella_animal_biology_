import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { UmbrellaLogo } from "@/components/umbrella/logo";
import { ThemeToggle } from "@/components/umbrella/theme-toggle";
import { useUmbrella } from "@/lib/umbrella-store";

const TITLE = "Sign in — Umbrella research platform";
const DESCRIPTION =
  "Sign in to Umbrella to continue your multi-agent research conversations across genomics, biodiversity, and literature.";

export const Route = createFileRoute("/signin")({
  head: () => ({
    meta: [
      { title: TITLE },
      { name: "description", content: DESCRIPTION },
      { property: "og:title", content: TITLE },
      { property: "og:description", content: DESCRIPTION },
    ],
  }),
  component: SignIn,
});

function SignIn() {
  const { signIn } = useUmbrella();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="flex items-center justify-between px-5 py-5">
        <Link to="/">
          <UmbrellaLogo />
        </Link>
        <ThemeToggle />
      </header>

      <main className="flex flex-1 items-center justify-center px-5 pb-16">
        <div className="w-full max-w-sm">
          <h1 className="text-2xl font-semibold">Welcome back</h1>
          <p className="mt-1.5 text-sm text-muted-foreground">
            Interface preview — any details will do.
          </p>

          <form
            className="mt-8 space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              signIn(email || "researcher@umbrella.ai");
              navigate({ to: "/chat" });
            }}
          >
            <div className="space-y-1.5">
              <Label htmlFor="email">Email address</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@lab.edu"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
              />
            </div>
            <Button type="submit" className="w-full">
              Sign In
            </Button>
          </form>

          <p className="mt-6 text-center text-sm text-muted-foreground">
            New to Umbrella?{" "}
            <Link to="/signup" className="font-medium text-primary underline underline-offset-4">
              Get started
            </Link>
          </p>
        </div>
      </main>
    </div>
  );
}