import { cn } from "@/lib/utils";

export function UmbrellaMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 48 48"
      aria-hidden="true"
      className={cn("h-7 w-7 text-primary", className)}
      fill="currentColor"
    >
      <path d="M22.5 3 5.2 12.9l17.3 9.9V3Z" />
      <path d="m25.5 3 17.3 9.9-17.3 9.9V3Z" opacity=".92" />
      <path d="M4 15.5v19.9l17.3-9.9L4 15.5Z" opacity=".92" />
      <path d="m44 15.5-17.3 10L44 35.4V15.5Z" />
      <path d="M5.2 38.1 22.5 48V28.2L5.2 38.1Z" />
      <path d="M25.5 28.2V48l17.3-9.9-17.3-9.9Z" opacity=".92" />
      <circle cx="24" cy="25.5" r="3.2" fill="var(--color-background)" />
    </svg>
  );
}

export function UmbrellaLogo({
  className,
  detailed = false,
}: {
  className?: string;
  detailed?: boolean;
}) {
  return (
    <span className={cn("flex items-center gap-3", className)}>
      <UmbrellaMark className={detailed ? "size-12" : undefined} />
      <span className="flex flex-col">
        <span
          className={cn("font-display text-lg font-semibold", detailed && "text-3xl leading-none")}
        >
          Umbrella
        </span>
        {detailed && <span className="mt-1 text-sm font-medium text-primary">Animal BioHub</span>}
      </span>
    </span>
  );
}
