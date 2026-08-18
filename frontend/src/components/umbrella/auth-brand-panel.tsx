import { Dna, ShieldCheck } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";

import { Reveal } from "@/components/umbrella/landing/reveal";
import { UmbrellaLogo } from "./logo";

const EASE = [0.22, 1, 0.36, 1] as const;

const headlineVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.12, delayChildren: 0.15 } },
};

const lineVariants = {
  hidden: { opacity: 0, x: -22 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.55, ease: EASE } },
};

export function AuthBrandPanel() {
  const reduceMotion = useReducedMotion();

  return (
    <section className="auth-brand-panel" aria-label="Umbrella Animal BioHub">
      <Reveal as="fade-up" eager duration={0.55}>
        <UmbrellaLogo detailed />
      </Reveal>
      <div className="auth-brand-copy">
        <motion.h2
          initial={reduceMotion ? false : "hidden"}
          animate="visible"
          variants={headlineVariants}
        >
          <motion.span className="block" variants={lineVariants}>
            One platform.
          </motion.span>
          <motion.span className="block" variants={lineVariants}>
            All animal genomics.
          </motion.span>
          <motion.span className="block" variants={lineVariants}>
            <span>Powered by AI.</span>
          </motion.span>
        </motion.h2>
        <motion.div
          className="auth-brand-rule"
          style={{ transformOrigin: "left" }}
          initial={reduceMotion ? false : { scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.6, delay: 0.55, ease: EASE }}
        />
        <Reveal as="fade-up" eager duration={0.55} delay={0.62}>
          <p>
            Access powerful AI agents for genome analysis, trait discovery, DNA reconstruction,
            biodiversity insights, and scientific literature, all in one conversational platform.
          </p>
        </Reveal>
      </div>
      <motion.div
        className="auth-brand-detail"
        aria-hidden="true"
        initial={reduceMotion ? false : { opacity: 0, scale: 0.75, rotate: -8 }}
        animate={{ opacity: 1, scale: 1, rotate: 0 }}
        transition={{ duration: 0.65, delay: 0.7, ease: EASE }}
      >
        <motion.div
          animate={reduceMotion ? undefined : { y: [0, -10, 0], rotate: [0, 4, 0, -4, 0] }}
          transition={{ duration: 9, repeat: Infinity, ease: "easeInOut", delay: 1.3 }}
        >
          <Dna />
        </motion.div>
      </motion.div>
    </section>
  );
}

export function AuthTrustNote() {
  return (
    <Reveal as="fade-up" duration={0.55} amount={0.4}>
      <div className="auth-trust-note">
        <ShieldCheck aria-hidden="true" />
        <span>
          <strong>Secure · Private · Research Ready</strong>
          <small>Your data is protected with enterprise-grade security.</small>
        </span>
      </div>
    </Reveal>
  );
}
