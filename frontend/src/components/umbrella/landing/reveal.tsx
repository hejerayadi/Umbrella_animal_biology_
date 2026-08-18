import { motion, useReducedMotion, type Variants } from "framer-motion";
import type { ReactNode } from "react";

const EASE = [0.22, 1, 0.36, 1] as const;

const VARIANTS: Record<string, Variants> = {
  "fade-up": {
    hidden: { opacity: 0, y: 26 },
    visible: { opacity: 1, y: 0 },
  },
  "fade-in": {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
  },
  "zoom-in": {
    hidden: { opacity: 0, scale: 0.9 },
    visible: { opacity: 1, scale: 1 },
  },
  "zoom-out": {
    hidden: { opacity: 0, scale: 1.08 },
    visible: { opacity: 1, scale: 1 },
  },
  "slide-left": {
    hidden: { opacity: 0, x: 36 },
    visible: { opacity: 1, x: 0 },
  },
  "slide-right": {
    hidden: { opacity: 0, x: -36 },
    visible: { opacity: 1, x: 0 },
  },
};

type RevealVariant = keyof typeof VARIANTS;

export function Reveal({
  children,
  as = "fade-up",
  delay = 0,
  duration = 0.6,
  className,
  once = false,
  amount = 0.3,
  eager = false,
}: {
  children: ReactNode;
  as?: RevealVariant;
  delay?: number;
  duration?: number;
  className?: string;
  once?: boolean;
  amount?: number;
  eager?: boolean;
}) {
  const reduceMotion = useReducedMotion();

  return (
    <motion.div
      className={className}
      initial={reduceMotion ? false : "hidden"}
      animate={eager ? "visible" : undefined}
      whileInView={eager ? undefined : "visible"}
      viewport={{ once, amount }}
      variants={VARIANTS[as]}
      transition={reduceMotion ? { duration: 0 } : { duration, delay, ease: EASE }}
    >
      {children}
    </motion.div>
  );
}

export function Stagger({
  children,
  className,
  stagger = 0.09,
  delay = 0,
  once = false,
  amount = 0.2,
  eager = false,
}: {
  children: ReactNode;
  className?: string;
  stagger?: number;
  delay?: number;
  once?: boolean;
  amount?: number;
  eager?: boolean;
}) {
  const reduceMotion = useReducedMotion();

  return (
    <motion.div
      className={className}
      initial={reduceMotion ? false : "hidden"}
      animate={eager ? "visible" : undefined}
      whileInView={eager ? undefined : "visible"}
      viewport={{ once, amount }}
      variants={{
        hidden: {},
        visible: { transition: { staggerChildren: stagger, delayChildren: delay } },
      }}
    >
      {children}
    </motion.div>
  );
}

export function StaggerItem({
  children,
  className,
  as = "fade-up",
}: {
  children: ReactNode;
  className?: string;
  as?: RevealVariant;
}) {
  return (
    <motion.div
      className={className}
      variants={VARIANTS[as]}
      transition={{ duration: 0.55, ease: EASE }}
    >
      {children}
    </motion.div>
  );
}

export function TypingText({ text, className }: { text: string; className?: string }) {
  const reduceMotion = useReducedMotion();

  if (reduceMotion) return <span className={className}>{text}</span>;

  return (
    <motion.span
      className={className}
      aria-label={text}
      initial="hidden"
      animate="visible"
      variants={{
        hidden: {},
        visible: { transition: { delayChildren: 0.12, staggerChildren: 0.045 } },
      }}
    >
      <span aria-hidden="true">
        {Array.from(text).map((character, index) => (
          <motion.span
            key={`${character}-${index}`}
            className="inline-block"
            variants={{
              hidden: { opacity: 0, y: 10, filter: "blur(5px)" },
              visible: { opacity: 1, y: 0, filter: "blur(0px)" },
            }}
            transition={{ duration: 0.24, ease: EASE }}
          >
            {character}
          </motion.span>
        ))}
      </span>
      <motion.span
        aria-hidden="true"
        className="landing-type-caret"
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 1, 1, 0] }}
        transition={{ delay: 0.1, duration: 0.65, repeat: 2, times: [0, 0.1, 0.7, 1] }}
      />
    </motion.span>
  );
}
