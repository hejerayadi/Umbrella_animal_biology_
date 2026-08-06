import { cn } from "@/lib/utils";

export function UmbrellaMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      aria-hidden="true"
      className={cn("h-7 w-7 text-primary", className)}
      fill="none"
    >
      <path
        d="M16 4C9.4 4 4 9.4 4 16h24C28 9.4 22.6 4 16 4Z"
        fill="currentColor"
        opacity="0.95"
      />
      <path
        d="M16 16v9a3 3 0 0 0 3 3 3 3 0 0 0 3-3"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      <circle cx="16" cy="16" r="2.1" fill="currentColor" />
      <path d="M4 16h24" stroke="currentColor" strokeWidth="1.6" opacity="0.35" />
    </svg>
  );
}

export function UmbrellaLogo({ className }: { className?: string }) {
  return (
    <span className={cn("flex items-center gap-2", className)}>
      <UmbrellaMark />
      <span className="font-display text-lg font-semibold tracking-tight">Umbrella</span>
    </span>
  );
}