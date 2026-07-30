import { Link, createFileRoute } from "@tanstack/react-router";
import {
  BookOpen,
  Dna,
  Leaf,
  Network,
  ScanSearch,
  Sparkle,
  TreeDeciduous,
  Boxes,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { UmbrellaLogo } from "@/components/umbrella/logo";
import { ThemeToggle } from "@/components/umbrella/theme-toggle";

const TITLE = "Umbrella — AI research ecosystem for genomics & biodiversity";
const DESCRIPTION =
  "Umbrella is a multi-agent AI platform for genomics, biodiversity, evolution, trait discovery, protein visualization, and scientific literature analysis.";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: TITLE },
      { name: "description", content: DESCRIPTION },
      { property: "og:title", content: TITLE },
      { property: "og:description", content: DESCRIPTION },
    ],
  }),
  component: Landing,
});

const CAPABILITIES = [
  {
    icon: Dna,
    title: "Genome Analysis & Reconstruction",
    body: "Assemble, scaffold, and interrogate genomes from raw sequence input.",
  },
  {
    icon: Leaf,
    title: "Biodiversity Intelligence",
    body: "Track richness, abundance, and ecosystem shifts across observation datasets.",
  },
  {
    icon: TreeDeciduous,
    title: "Evolutionary Research Support",
    body: "Reason over phylogenies, divergence timing, and comparative lineage signals.",
  },
  {
    icon: ScanSearch,
    title: "Trait Discovery",
    body: "Surface candidate loci and phenotype associations across related species.",
  },
  {
    icon: Boxes,
    title: "Protein Visualization",
    body: "Compare folds, pockets, and structural variants with structural context.",
  },
  {
    icon: BookOpen,
    title: "Scientific Literature Exploration",
    body: "Screen, synthesise, and cross-reference the publication record at speed.",
  },
  {
    icon: Network,
    title: "Multi-Agent AI Orchestration",
    body: "A planner routes each request to the specialist agents that fit it best.",
  },
];

const STEPS = [
  {
    step: "01",
    title: "You ask in plain language",
    body: "A single conversational interface replaces a pile of disconnected tools and scripts.",
  },
  {
    step: "02",
    title: "Specialist agents collaborate",
    body: "Genome, biodiversity, trait, protein, and literature agents work the problem in parallel.",
  },
  {
    step: "03",
    title: "The system plans and executes",
    body: "An orchestrator decomposes the task, sequences the agents, and reconciles their findings.",
  },
];

function Landing() {
  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-20 border-b border-border/70 bg-background/85 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5">
          <UmbrellaLogo />
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Button variant="ghost" asChild className="hidden sm:inline-flex">
              <Link to="/signin">Sign In</Link>
            </Button>
            <Button asChild>
              <Link to="/signup">Get Started</Link>
            </Button>
          </div>
        </div>
      </header>

      <main>
        <section className="relative overflow-hidden hero-glow">
          <div className="pointer-events-none absolute inset-0 grid-faint [mask-image:radial-gradient(70%_60%_at_50%_0%,black,transparent)]" />
          <div className="relative mx-auto max-w-3xl px-5 pb-24 pt-24 text-center">
            <span className="inline-flex items-center gap-2 rounded-full border border-primary/25 bg-accent px-3 py-1 text-xs font-medium text-accent-foreground">
              <Sparkle className="h-3.5 w-3.5" />
              Multi-agent research orchestration
            </span>
            <h1 className="mt-7 text-balance text-5xl font-semibold leading-[1.05] sm:text-6xl">
              Umbrella
            </h1>
            <p className="mx-auto mt-5 max-w-2xl text-balance text-lg text-foreground/80 sm:text-xl">
              An AI-powered research ecosystem for genomics, biodiversity, and scientific discovery.
            </p>
            <p className="mx-auto mt-5 max-w-2xl text-pretty text-[0.98rem] leading-7 text-muted-foreground">
              Umbrella is a multi-agent AI platform that helps researchers analyse biological data,
              explore the scientific literature, and generate insight across genomics, evolution,
              traits, and protein structures — all from one conversation.
            </p>
            <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
              <Button size="lg" asChild>
                <Link to="/signup">Get Started</Link>
              </Button>
              <Button size="lg" variant="outline" asChild>
                <Link to="/signin">Sign In</Link>
              </Button>
            </div>
          </div>
        </section>

        <section className="border-t border-border bg-card/40">
          <div className="mx-auto max-w-6xl px-5 py-20">
            <h2 className="text-3xl font-semibold">Key capabilities</h2>
            <p className="mt-2 max-w-xl text-sm text-muted-foreground">
              Seven specialist domains, one coordinated system.
            </p>
            <div className="mt-10 grid gap-px overflow-hidden rounded-xl border border-border bg-border sm:grid-cols-2 lg:grid-cols-3">
              {CAPABILITIES.map((capability, i) => (
                <article
                  key={capability.title}
                  className={
                    i === CAPABILITIES.length - 1
                      ? "bg-background p-6 sm:col-span-2 lg:col-span-3"
                      : "bg-background p-6"
                  }
                >
                  <capability.icon className="h-5 w-5 text-primary" />
                  <h3 className="mt-4 text-base font-semibold">{capability.title}</h3>
                  <p className="mt-2 text-sm leading-6 text-muted-foreground">{capability.body}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="border-t border-border">
          <div className="mx-auto max-w-6xl px-5 py-20">
            <h2 className="text-3xl font-semibold">How it works</h2>
            <div className="mt-10 grid gap-10 md:grid-cols-3">
              {STEPS.map((item) => (
                <div key={item.step} className="border-t-2 border-primary pt-5">
                  <span className="font-[family-name:var(--font-mono-custom)] text-xs text-primary">
                    {item.step}
                  </span>
                  <h3 className="mt-2 text-lg font-semibold">{item.title}</h3>
                  <p className="mt-2 text-sm leading-6 text-muted-foreground">{item.body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="border-t border-border bg-card/40">
          <div className="mx-auto max-w-3xl px-5 py-20 text-center">
            <h2 className="text-3xl font-semibold">Start your first research thread</h2>
            <p className="mx-auto mt-3 max-w-xl text-sm text-muted-foreground">
              Set up your research profile in two steps and open the conversation.
            </p>
            <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
              <Button size="lg" asChild>
                <Link to="/signup">Get Started</Link>
              </Button>
              <Button size="lg" variant="outline" asChild>
                <Link to="/signin">Sign In</Link>
              </Button>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t border-border py-8">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 text-xs text-muted-foreground">
          <UmbrellaLogo className="opacity-80" />
          <span>Research preview · interface only</span>
        </div>
      </footer>
    </div>
  );
}
