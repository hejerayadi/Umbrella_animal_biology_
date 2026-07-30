import { ArrowUp, Mic, Paperclip } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";

export function ChatComposer({
  onSend,
  disabled,
  focusKey,
}: {
  onSend: (value: string) => void;
  disabled?: boolean;
  focusKey?: string;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    ref.current?.focus();
  }, [focusKey, disabled]);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    ref.current?.focus();
  };

  return (
    <div className="rounded-2xl border border-border bg-card p-2 shadow-[var(--shadow-soft)]">
      <Textarea
        ref={ref}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        rows={2}
        placeholder="Ask Umbrella about genomes, traits, species, structures, or literature…"
        className="max-h-48 min-h-[56px] resize-none border-0 bg-transparent px-2 py-2 text-[0.95rem] shadow-none focus-visible:ring-0 dark:bg-transparent"
      />
      <div className="flex items-center justify-between px-1 pt-1">
        <div className="flex items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Attach file"
            onClick={() => toast("Attachments arrive with the backend phase.")}
          >
            <Paperclip className="h-4 w-4" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Voice input"
            onClick={() => toast("Voice input is a placeholder for now.")}
          >
            <Mic className="h-4 w-4" />
          </Button>
          <span className="ml-1 hidden text-[0.7rem] text-muted-foreground sm:inline">
            Enter to send · Shift + Enter for a new line
          </span>
        </div>
        <Button
          type="button"
          size="icon"
          onClick={submit}
          disabled={disabled || !value.trim()}
          aria-label="Send message"
          className="size-9 rounded-full"
        >
          <ArrowUp className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}