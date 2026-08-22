import { cn } from "@/lib/utils";

/**
 * Shared decorative science/biology watermark (DNA helix, molecule graph,
 * animal silhouettes, dotted wave) reused across landing sections. Each
 * placement crops a different region of the same source image via
 * object-position so the page doesn't need multiple bespoke assets.
 */
function ScienceArt({
  className,
  imageClassName,
  flip = false,
}: {
  className?: string;
  imageClassName?: string;
  flip?: boolean;
}) {
  return (
    <div
      className={cn("pointer-events-none absolute select-none overflow-hidden", className)}
      aria-hidden="true"
    >
      <img
        src="/umbrella-auth-science.png"
        alt=""
        className={cn("h-full w-full object-cover", flip && "-scale-x-100", imageClassName)}
      />
    </div>
  );
}

export function HeroDecoration() {
  return (
    <ScienceArt
      className="inset-0 opacity-[0.55] dark:opacity-[0.16]"
      imageClassName="object-[85%_20%]"
    />
  );
}

export function FieldNotesDecoration() {
  return (
    <ScienceArt
      className="bottom-0 left-0 h-64 w-[36rem] opacity-[0.5] dark:opacity-[0.14] [mask-image:linear-gradient(to_top,black,transparent)]"
      imageClassName="object-[8%_78%]"
    />
  );
}

export function CtaDecoration() {
  return (
    <ScienceArt
      className="inset-y-0 right-0 w-[30rem] opacity-[0.45] dark:opacity-[0.12] [mask-image:linear-gradient(to_left,black,transparent)]"
      imageClassName="object-[95%_35%]"
      flip
    />
  );
}
