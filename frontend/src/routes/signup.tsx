import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import { ArrowLeft, ArrowRight, Check } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { UmbrellaLogo } from "@/components/umbrella/logo";
import { ThemeToggle } from "@/components/umbrella/theme-toggle";
import { useUmbrella } from "@/lib/umbrella-store";
import {
  AI_CAPABILITIES,
  USER_ROLES,
  type AiCapability,
  type UserRole,
} from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

const TITLE = "Get started — Umbrella research onboarding";
const DESCRIPTION =
  "Create your Umbrella research profile: tell the multi-agent system your role, goals, and the AI capabilities you care about.";

export const Route = createFileRoute("/signup")({
  head: () => ({
    meta: [
      { title: TITLE },
      { name: "description", content: DESCRIPTION },
      { property: "og:title", content: TITLE },
      { property: "og:description", content: DESCRIPTION },
    ],
  }),
  component: SignUp,
});

function SignUp() {
  const { signUp } = useUmbrella();
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<UserRole>("Researcher");
  const [purpose, setPurpose] = useState("");
  const [mainInterest, setMainInterest] = useState("");
  const [goals, setGoals] = useState("");
  const [interests, setInterests] = useState<AiCapability[]>([]);

  const toggleInterest = (capability: AiCapability) =>
    setInterests((prev) =>
      prev.includes(capability) ? prev.filter((c) => c !== capability) : [...prev, capability],
    );

  const stepOneValid = name.trim() && email.trim() && password.trim();

  const finish = () => {
    signUp({
      name: name.trim(),
      email: email.trim(),
      role,
      purpose,
      mainInterest,
      goals,
      researchInterests: interests,
    });
    navigate({ to: "/chat" });
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="flex items-center justify-between px-5 py-5">
        <Link to="/">
          <UmbrellaLogo />
        </Link>
        <ThemeToggle />
      </header>

      <main className="flex flex-1 justify-center px-5 pb-20">
        <div className="w-full max-w-xl">
          <div className="flex items-center gap-3">
            {[1, 2].map((n) => (
              <div key={n} className="flex flex-1 items-center gap-2">
                <span
                  className={cn(
                    "flex size-6 items-center justify-center rounded-full border text-xs font-medium",
                    step >= n
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border text-muted-foreground",
                  )}
                >
                  {step > n ? <Check className="h-3 w-3" /> : n}
                </span>
                <span className="h-px flex-1 bg-border" />
              </div>
            ))}
          </div>

          {step === 1 ? (
            <section className="mt-8 animate-fade-up">
              <h1 className="text-2xl font-semibold">Basic information</h1>
              <p className="mt-1.5 text-sm text-muted-foreground">
                Step 1 of 2 — who is joining the research ecosystem.
              </p>

              <form
                className="mt-8 space-y-4"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (stepOneValid) setStep(2);
                }}
              >
                <div className="space-y-1.5">
                  <Label htmlFor="name">Full name</Label>
                  <Input
                    id="name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Dr. Amara Sørensen"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="email">Email address</Label>
                  <Input
                    id="email"
                    type="email"
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
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="role">Role</Label>
                  <Select value={role} onValueChange={(v) => setRole(v as UserRole)}>
                    <SelectTrigger id="role">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {USER_ROLES.map((r) => (
                        <SelectItem key={r} value={r}>
                          {r}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <Button type="submit" disabled={!stepOneValid} className="w-full gap-2">
                  Continue
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </form>

              <p className="mt-6 text-center text-sm text-muted-foreground">
                Already have an account?{" "}
                <Link to="/signin" className="font-medium text-primary underline underline-offset-4">
                  Sign in
                </Link>
              </p>
            </section>
          ) : (
            <section className="mt-8 animate-fade-up">
              <h1 className="text-2xl font-semibold">Your research goals</h1>
              <p className="mt-1.5 text-sm text-muted-foreground">
                Step 2 of 2 — this shapes how agents are selected for you.
              </p>

              <form
                className="mt-8 space-y-5"
                onSubmit={(e) => {
                  e.preventDefault();
                  finish();
                }}
              >
                <div className="space-y-1.5">
                  <Label htmlFor="purpose">Why are you using Umbrella?</Label>
                  <Textarea
                    id="purpose"
                    rows={2}
                    value={purpose}
                    onChange={(e) => setPurpose(e.target.value)}
                    placeholder="To accelerate comparative genomics across endangered species."
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="interest">What is your main research interest?</Label>
                  <Input
                    id="interest"
                    value={mainInterest}
                    onChange={(e) => setMainInterest(e.target.value)}
                    placeholder="Trait evolution in vertebrates"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="goals">What are you hoping to accomplish?</Label>
                  <Textarea
                    id="goals"
                    rows={2}
                    value={goals}
                    onChange={(e) => setGoals(e.target.value)}
                    placeholder="Publish a cross-species trait atlas."
                  />
                </div>

                <div className="space-y-2">
                  <Label>Which AI capabilities interest you most?</Label>
                  <div className="flex flex-wrap gap-2">
                    {AI_CAPABILITIES.map((capability) => {
                      const selected = interests.includes(capability);
                      return (
                        <button
                          key={capability}
                          type="button"
                          onClick={() => toggleInterest(capability)}
                          aria-pressed={selected}
                          className={cn(
                            "rounded-full border px-3.5 py-1.5 text-sm transition-colors",
                            selected
                              ? "border-primary bg-primary text-primary-foreground"
                              : "border-border text-muted-foreground hover:border-primary/40 hover:text-foreground",
                          )}
                        >
                          {capability}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button type="button" variant="outline" onClick={() => setStep(1)} className="gap-2">
                    <ArrowLeft className="h-4 w-4" />
                    Back
                  </Button>
                  <Button type="submit" className="flex-1">
                    Enter Umbrella
                  </Button>
                </div>
              </form>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}