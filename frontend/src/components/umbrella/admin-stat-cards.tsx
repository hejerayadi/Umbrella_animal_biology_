import { motion, useReducedMotion } from "framer-motion";
import type { LucideIcon } from "lucide-react";

const EASE = [0.22, 1, 0.36, 1] as const;

export interface AdminStatCard {
  label: string;
  value: number | string;
  note: string;
  icon: LucideIcon;
  tone: "primary" | "green" | "amber" | "blue" | "red";
}

export function AdminStatCards({
  cards,
  loading = false,
}: {
  cards: AdminStatCard[];
  loading?: boolean;
}) {
  const reduceMotion = useReducedMotion();

  return (
    <motion.div
      className="admin-stat-grid"
      initial="hidden"
      animate="visible"
      variants={{
        hidden: {},
        visible: { transition: { staggerChildren: 0.07 } },
      }}
    >
      {cards.map((card) => (
        <motion.article
          key={card.label}
          className="admin-stat-card"
          variants={{
            hidden: reduceMotion ? {} : { opacity: 0, y: 14, scale: 0.98 },
            visible: { opacity: 1, y: 0, scale: 1 },
          }}
          transition={{ duration: 0.4, ease: EASE }}
          whileHover={reduceMotion ? undefined : { y: -3 }}
        >
          <span className={`admin-stat-icon is-${card.tone}`}>
            <card.icon />
          </span>
          <div>
            <p>{card.label}</p>
            <motion.strong
              key={card.value}
              initial={reduceMotion ? false : { opacity: 0, scale: 0.86 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              {loading ? "--" : card.value}
            </motion.strong>
            <small>{card.note}</small>
          </div>
        </motion.article>
      ))}
    </motion.div>
  );
}
