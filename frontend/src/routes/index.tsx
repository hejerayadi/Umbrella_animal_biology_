import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import {
  ArrowRight,
  ArrowUp,
  Atom,
  BarChart3,
  BookOpen,
  Boxes,
  CheckCircle2,
  CircleDot,
  Dna,
  Leaf,
  MapPin,
  Network,
  Paperclip,
  ScanSearch,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TreeDeciduous,
  Umbrella as UmbrellaIcon,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { useRef, useState, type FormEvent } from "react";

import { Button } from "@/components/ui/button";
import {
  CtaDecoration,
  FieldNotesDecoration,
  HeroDecoration,
} from "@/components/umbrella/landing/decorations";
import { Reveal, Stagger, StaggerItem, TypingText } from "@/components/umbrella/landing/reveal";
import { UmbrellaLogo, UmbrellaMark } from "@/components/umbrella/logo";
import { cn } from "@/lib/utils";

const TITLE = "Umbrella - AI research ecosystem for genomics & biodiversity";
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
    code: "ACC. 01",
    icon: Dna,
    title: "Genome Analysis & Reconstruction",
    body: "Assemble, scaffold, and interrogate genomes from raw sequence input.",
  },
  {
    code: "ACC. 02",
    icon: Leaf,
    title: "Biodiversity Intelligence",
    body: "Track richness, abundance, and ecosystem shifts across observation datasets.",
  },
  {
    code: "ACC. 03",
    icon: Network,
    title: "Evolutionary Research Support",
    body: "Reason over phylogenies, divergence timing, and comparative lineage signals.",
  },
  {
    code: "ACC. 04",
    icon: BarChart3,
    title: "Trait Discovery",
    body: "Surface candidate loci and phenotype associations across related species.",
  },
  {
    code: "ACC. 05",
    icon: Atom,
    title: "Protein Visualization",
    body: "Compare folds, pockets, and structural variants with structural context.",
  },
  {
    code: "ACC. 06",
    icon: BookOpen,
    title: "Scientific Literature Exploration",
    body: "Screen, synthesise, and cross-reference the publication record at speed.",
  },
  {
    code: "ACC. 07",
    icon: Boxes,
    title: "Multi-Agent AI Orchestration",
    body: "A planner routes each request to the specialist agents that fit it best.",
  },
];

const PROCESS = [
  {
    step: "01",
    tag: "Intake",
    icon: ScanSearch,
    title: "You ask in plain language",
    body: "A single conversational interface replaces a pile of disconnected tools and scripts.",
  },
  {
    step: "02",
    tag: "Delegation",
    icon: Network,
    title: "Specialist agents collaborate",
    body: "Genome, biodiversity, trait, protein, and literature agents work the problem in parallel.",
  },
  {
    step: "03",
    tag: "Synthesis",
    icon: UmbrellaIcon,
    title: "The system plans and executes",
    body: "An orchestrator decomposes the task, sequences the agents, and reconciles their findings.",
  },
];

const AGENT_ACTIVITY = [
  { icon: Dna, name: "Genome Agent", detail: "scaffold reconstruction complete - N50 4.2 Mb" },
  {
    icon: BookOpen,
    name: "Literature Agent",
    detail: "cross-referenced 212 papers on clinal variation",
  },
  { icon: Atom, name: "Protein Agent", detail: "ranked 3 candidate folds by pocket similarity" },
  {
    icon: Leaf,
    name: "Biodiversity Agent",
    detail: "flagged abundance drop across 4 survey sites",
  },
];

const METRICS = [
  { icon: Network, value: "7", label: "specialist domains" },
  { icon: BookOpen, value: "212", label: "papers" },
  { icon: MapPin, value: "4", label: "survey sites" },
  { icon: Boxes, value: "3", label: "candidate folds" },
];

const FIELD_NOTES = [
  {
    icon: Dna,
    name: "Genome Agent",
    detail: "scaffold reconstruction complete - N50 4.2 Mb",
    time: "2m ago",
  },
  {
    icon: BookOpen,
    name: "Literature Agent",
    detail: "cross-referenced 212 papers on clinal variation",
    time: "6m ago",
  },
  {
    icon: Atom,
    name: "Protein Agent",
    detail: "ranked 3 candidate folds by pocket similarity",
    time: "11m ago",
  },
  {
    icon: Leaf,
    name: "Biodiversity Agent",
    detail: "flagged abundance drop across 4 survey sites",
    time: "18m ago",
  },
  {
    icon: BarChart3,
    name: "Trait Agent",
    detail: "surfaced 9 candidate loci for pigmentation",
    time: "24m ago",
  },
  {
    icon: TreeDeciduous,
    name: "Evolution Agent",
    detail: "re-estimated divergence time at 3.1 Mya",
    time: "31m ago",
  },
  {
    icon: Network,
    name: "Orchestrator",
    detail: "reconciled 4 agent outputs into one thread",
    time: "38m ago",
  },
];

const MOTION_EASE = [0.22, 1, 0.36, 1] as const;

function Landing() {
  const navigate = useNavigate();
  const heroRef = useRef<HTMLElement>(null);
  const [ctaQuery, setCtaQuery] = useState("");
  const [activeCapability, setActiveCapability] = useState(0);
  const activeCapabilityData = CAPABILITIES[activeCapability];
  const ActiveCapabilityIcon = activeCapabilityData.icon;
  const reduceMotion = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  });
  const heroCopyY = useTransform(scrollYProgress, [0, 1], [0, 90]);
  const heroPanelY = useTransform(scrollYProgress, [0, 1], [0, 48]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.82], [1, 0.2]);
  const heroScale = useTransform(scrollYProgress, [0, 1], [1, 0.94]);

  const handleCtaSubmit = (event: FormEvent) => {
    event.preventDefault();
    navigate({ to: "/signup" });
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="landing-page">
      <header className="landing-header-shell">
        <div className="landing-header">
          <Link to="/" className="landing-brand-link" aria-label="Umbrella home">
            <UmbrellaLogo detailed />
            <UmbrellaIcon className="landing-brand-umbrella" aria-hidden="true" />
          </Link>
          <nav className="landing-nav" aria-label="Primary navigation">
            <a href="#capabilities">Capabilities</a>
            <a href="#how-it-works">How it works</a>
            <a href="#field-notes">Field notes</a>
          </nav>
          <div className="landing-header-actions">
            <Button variant="outline" asChild>
              <Link to="/signin">Sign in</Link>
            </Button>
            <Button className="landing-primary-button" asChild>
              <Link to="/signup">
                Get started <ArrowRight />
              </Link>
            </Button>
          </div>
        </div>
      </header>

      <main>
        <section ref={heroRef} className="landing-hero">
          <HeroDecoration />
          <div className="landing-container landing-hero-grid">
            <motion.div
              className="landing-hero-copy"
              style={reduceMotion ? undefined : { y: heroCopyY, opacity: heroOpacity }}
            >
              <Reveal as="slide-right" duration={0.7} eager>
                <div className="landing-kicker">
                  <span>UMB · 001</span>
                  <span>Multi-agent research system</span>
                </div>
              </Reveal>
              <h1>
                <TypingText text="Umbrella" />
              </h1>
              <Reveal as="slide-right" delay={0.3} duration={0.55} eager>
                <p className="landing-hero-subtitle">
                  An <em>AI-powered</em> research ecosystem for genomics, biodiversity, and
                  scientific discovery.
                </p>
              </Reveal>
              <Reveal as="fade-up" delay={0.42} duration={0.55} eager>
                <p className="landing-hero-body">
                  One conversation, coordinated across specialist agents - genome reconstruction,
                  biodiversity intelligence, evolutionary reasoning, trait discovery, protein
                  structure, and the published literature - planned and executed as a single
                  research thread.
                </p>
              </Reveal>
              <Reveal as="fade-up" delay={0.54} duration={0.55} eager>
                <div className="landing-hero-actions">
                  <Button size="lg" className="landing-primary-button" asChild>
                    <Link to="/signup">
                      Get started <ArrowRight />
                    </Link>
                  </Button>
                  <Button size="lg" variant="outline" asChild>
                    <Link to="/signin">Sign in</Link>
                  </Button>
                </div>
              </Reveal>
              <Reveal as="zoom-in" delay={0.64} duration={0.5} eager>
                <div className="landing-hero-trust">
                  <ShieldCheck />
                  <span>One research thread</span>
                  <i />
                  <span>7 specialist domains</span>
                </div>
              </Reveal>
            </motion.div>

            <motion.div
              className="landing-hero-visual"
              style={
                reduceMotion ? undefined : { y: heroPanelY, opacity: heroOpacity, scale: heroScale }
              }
              initial={reduceMotion ? false : { opacity: 0, x: 72, scale: 0.9 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              transition={{ delay: 0.12, duration: 0.7, ease: MOTION_EASE }}
            >
              <div className="landing-dashboard-back" aria-hidden="true" />
              <div className="landing-dashboard-rail" aria-hidden="true">
                {[CircleDot, BookOpen, Network, Atom, SlidersHorizontal].map((Icon, index) => (
                  <Icon key={index} />
                ))}
              </div>
              <div className="landing-agent-panel">
                <div className="landing-panel-heading">
                  <strong>Agent activity</strong>
                  <span>
                    <i /> Live
                  </span>
                </div>
                <Stagger className="landing-agent-list" delay={0.28} stagger={0.07} eager>
                  {AGENT_ACTIVITY.map((item) => (
                    <StaggerItem key={item.name} as="slide-left">
                      <div className="landing-agent-row">
                        <span className="landing-agent-icon">
                          <item.icon />
                        </span>
                        <span>
                          <strong>{item.name}</strong>
                          <small>{item.detail}</small>
                        </span>
                        <CheckCircle2 />
                      </div>
                    </StaggerItem>
                  ))}
                </Stagger>
              </div>
              <Reveal
                as="zoom-in"
                delay={0.62}
                duration={0.5}
                className="landing-metric-panel"
                eager
              >
                {METRICS.map((metric) => (
                  <div key={metric.label}>
                    <metric.icon />
                    <strong>{metric.value}</strong>
                    <span>{metric.label}</span>
                  </div>
                ))}
              </Reveal>
            </motion.div>
          </div>
        </section>

        <section id="capabilities" className="landing-section landing-capabilities">
          <div className="landing-container">
            <SectionIntro
              eyebrow="Collection - 7 specialist domains"
              title="What lives under the umbrella"
              body="Every request is routed to the specialists built for it, then reconciled back into one answer."
            />
            <Reveal as="fade-up" delay={0.1} className="landing-capability-layout">
              <div
                className="landing-capability-list"
                role="tablist"
                aria-label="Specialist domains"
              >
                {CAPABILITIES.map((capability, index) => {
                  const isActive = index === activeCapability;
                  return (
                    <button
                      key={capability.code}
                      type="button"
                      role="tab"
                      aria-selected={isActive}
                      className={cn("landing-capability-row", isActive && "is-active")}
                      onMouseEnter={() => setActiveCapability(index)}
                      onFocus={() => setActiveCapability(index)}
                      onClick={() => setActiveCapability(index)}
                    >
                      <span className="landing-capability-row-code">{capability.code}</span>
                      <span className="landing-capability-row-icon">
                        <capability.icon />
                      </span>
                      <span className="landing-capability-row-title">{capability.title}</span>
                      <ArrowRight className="landing-capability-row-arrow" />
                    </button>
                  );
                })}
              </div>

              <div className="landing-capability-preview">
                <span className="landing-capability-preview-index">
                  {String(activeCapability + 1).padStart(2, "0")} /{" "}
                  {String(CAPABILITIES.length).padStart(2, "0")}
                </span>
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeCapabilityData.code}
                    className="landing-capability-preview-body"
                    initial={reduceMotion ? false : { opacity: 0, y: 14 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={reduceMotion ? undefined : { opacity: 0, y: -14 }}
                    transition={{ duration: 0.35, ease: MOTION_EASE }}
                  >
                    <span className="landing-capability-preview-watermark" aria-hidden="true">
                      <ActiveCapabilityIcon />
                    </span>
                    <span className="landing-card-code">{activeCapabilityData.code}</span>
                    <span className="landing-capability-icon">
                      <ActiveCapabilityIcon />
                    </span>
                    <h3>{activeCapabilityData.title}</h3>
                    <p>{activeCapabilityData.body}</p>
                  </motion.div>
                </AnimatePresence>
                <div className="landing-capability-progress">
                  {CAPABILITIES.map((capability, index) => (
                    <span
                      key={capability.code}
                      className={cn(
                        index === activeCapability && "is-active",
                        index < activeCapability && "is-done",
                      )}
                    />
                  ))}
                </div>
              </div>
            </Reveal>
            <Reveal as="fade-in" className="landing-scroll-note">
              <span>Hover or tap a domain to preview its scope</span>
              <ArrowRight />
            </Reveal>
          </div>
        </section>

        <section id="how-it-works" className="landing-section landing-process">
          <div className="landing-container landing-section-split">
            <SectionIntro
              eyebrow="Process"
              title="How it works"
              body="A single trunk of intent branches into specialist work, then folds back into one answer."
            />
            <Stagger className="landing-process-grid" stagger={0.16} amount={0.3}>
              {PROCESS.map((item, index) => (
                <StaggerItem key={item.step} as={index === 1 ? "zoom-in" : "fade-up"}>
                  <motion.article
                    className="landing-process-card"
                    whileHover={reduceMotion ? undefined : { y: -5 }}
                  >
                    <div className="landing-process-topline">
                      <span>{item.step}</span>
                      <small>{item.tag}</small>
                    </div>
                    <div className="landing-process-title">
                      <span>
                        <item.icon />
                      </span>
                      <h3>{item.title}</h3>
                    </div>
                    <p>{item.body}</p>
                  </motion.article>
                </StaggerItem>
              ))}
            </Stagger>
          </div>
        </section>

        <section id="field-notes" className="landing-section landing-field-notes">
          <FieldNotesDecoration />
          <div className="landing-container landing-section-split">
            <SectionIntro
              eyebrow="Field notes"
              title="Recent orchestration events"
              body="A running log of what the agents have been doing across active research threads."
            />
            <div className="landing-event-panel">
              {FIELD_NOTES.map((item, index) => (
                <Reveal
                  key={item.name}
                  as="slide-left"
                  delay={index * 0.065}
                  amount={0.15}
                  className={cn("landing-event-row", index === 0 && "is-current")}
                >
                  <span className="landing-event-dot" />
                  <span className="landing-event-icon">
                    <item.icon />
                  </span>
                  <strong>{item.name}</strong>
                  <span className="landing-event-detail">{item.detail}</span>
                  <time>{item.time}</time>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        <section className="landing-section landing-cta-section">
          <CtaDecoration />
          <div className="landing-container">
            <Reveal as="zoom-out" amount={0.35}>
              <div className="landing-cta-card">
                <div>
                  <span className="landing-eyebrow">Start here</span>
                  <h2>Open your first research thread</h2>
                  <p>
                    Set up your research profile in two steps, then hand the first question to the
                    system.
                  </p>
                  <div className="landing-cta-actions">
                    <Button className="landing-primary-button" asChild>
                      <Link to="/signup">
                        Get started <ArrowRight />
                      </Link>
                    </Button>
                    <Button variant="outline" asChild>
                      <Link to="/signin">Sign in</Link>
                    </Button>
                  </div>
                </div>
                <form className="landing-question-card" onSubmit={handleCtaSubmit}>
                  <label htmlFor="landing-question">Ask your research question</label>
                  <div>
                    <input
                      id="landing-question"
                      value={ctaQuery}
                      onChange={(event) => setCtaQuery(event.target.value)}
                      placeholder="e.g., Compare adaptive loci for drought tolerance across savannah mammals"
                    />
                    <button type="submit" aria-label="Open research profile">
                      <Send />
                    </button>
                  </div>
                  <footer>
                    <span>
                      <Paperclip /> Attach files
                    </span>
                    <span>
                      <Leaf /> Select organism (optional)
                    </span>
                    <span>
                      <SlidersHorizontal /> Advanced
                    </span>
                  </footer>
                </form>
              </div>
            </Reveal>
          </div>
        </section>
      </main>

      <footer className="landing-footer">
        <Reveal as="fade-up" amount={0.15} className="landing-container landing-footer-top">
          <div className="landing-footer-brand-col">
            <span className="landing-footer-brand">
              <UmbrellaMark /> Umbrella <UmbrellaIcon />
            </span>
            <p>
              A multi-agent AI research ecosystem coordinating genome analysis, biodiversity
              intelligence, evolutionary reasoning, and the published literature into one thread.
            </p>
            <span className="landing-footer-badge">
              <Sparkles /> Research preview · interface only
            </span>
          </div>

          <nav className="landing-footer-col" aria-label="Explore">
            <span className="landing-footer-col-title">Explore</span>
            <a href="#capabilities">Capabilities</a>
            <a href="#how-it-works">How it works</a>
            <a href="#field-notes">Field notes</a>
          </nav>

          <nav className="landing-footer-col" aria-label="Account">
            <span className="landing-footer-col-title">Account</span>
            <Link to="/signin">Sign in</Link>
            <Link to="/signup">Get started</Link>
          </nav>

          <div className="landing-footer-col landing-footer-domains">
            <span className="landing-footer-col-title">Specialist domains</span>
            <ul>
              <li>
                <Dna aria-hidden="true" /> Genome analysis
              </li>
              <li>
                <Leaf aria-hidden="true" /> Biodiversity
              </li>
              <li>
                <Atom aria-hidden="true" /> Protein structure
              </li>
              <li>
                <BookOpen aria-hidden="true" /> Literature
              </li>
            </ul>
          </div>
        </Reveal>

        <div className="landing-container landing-footer-bottom">
          <span>© 2026 Umbrella Animal BioHub. All rights reserved.</span>
          <button type="button" className="landing-footer-top-btn" onClick={scrollToTop}>
            Back to top <ArrowUp />
          </button>
        </div>
      </footer>
    </div>
  );
}

function SectionIntro({ eyebrow, title, body }: { eyebrow: string; title: string; body: string }) {
  return (
    <div className="landing-section-intro">
      <Reveal as="slide-right">
        <span className="landing-eyebrow">{eyebrow}</span>
      </Reveal>
      <Reveal as="fade-up" delay={0.06}>
        <h2>{title}</h2>
      </Reveal>
      <Reveal as="fade-up" delay={0.12}>
        <p>{body}</p>
      </Reveal>
    </div>
  );
}
