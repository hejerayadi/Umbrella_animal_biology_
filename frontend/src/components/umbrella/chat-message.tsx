import { useEffect, useState } from "react";

import { BiodiversityMap } from "@/components/umbrella/biodiversity-map";
import { GenomeChart } from "@/components/umbrella/genome-chart";
import { Markdown } from "@/components/umbrella/markdown";
import { UmbrellaMark } from "@/components/umbrella/logo";
import { ProteinViewer } from "@/components/umbrella/protein-viewer";
import { ReconstructionPanel } from "@/components/umbrella/reconstruction-panel";
import { RecognitionPanel } from "@/components/umbrella/recognition-panel";
import { UserAvatar } from "@/components/umbrella/user-avatar";
import { WritingPanel } from "@/components/umbrella/writing-panel";
import type { Message } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

function useTypedText(text: string, enabled: boolean) {
  const [shown, setShown] = useState(enabled ? "" : text);

  useEffect(() => {
    if (!enabled) {
      setShown(text);
      return;
    }
    let index = 0;
    setShown("");
    const interval = setInterval(() => {
      index = Math.min(text.length, index + 6);
      setShown(text.slice(0, index));
      if (index >= text.length) clearInterval(interval);
    }, 16);
    return () => clearInterval(interval);
  }, [text, enabled]);

  return { shown, done: shown.length >= text.length };
}

export function ChatMessage({
  message,
  userName,
  streaming = false,
}: {
  message: Message;
  userName: string;
  streaming?: boolean;
}) {
  const isUser = message.sender === "user";
  const { shown, done } = useTypedText(message.content, streaming && !isUser);
  // The backend evicts old uploads, so the URL can 404 in a long-lived
  // conversation. Hide the image rather than leaving a broken-image icon -
  // the message text is still perfectly readable without it.
  const [imageFailed, setImageFailed] = useState(false);
  const showImage = Boolean(message.imageUrl) && !imageFailed;
  // Same eviction story as the attachment above, for the generated
  // illustration on an assistant message.
  const [generatedFailed, setGeneratedFailed] = useState(false);
  const showGenerated = Boolean(message.generatedImageUrl) && !generatedFailed;

  return (
    <div className={cn("flex w-full gap-3 animate-fade-up", isUser && "justify-end")}>
      {!isUser && (
        <span className="mt-1 flex size-8 shrink-0 items-center justify-center rounded-md border border-border bg-card">
          <UmbrellaMark className="h-4 w-4" />
        </span>
      )}

      <div className={cn("min-w-0", isUser ? "max-w-[80%]" : "flex-1")}>
        {isUser ? (
          <div className="flex flex-col items-end gap-2">
            {showImage && (
              <a
                href={message.imageUrl}
                target="_blank"
                rel="noreferrer"
                className="block overflow-hidden rounded-2xl rounded-tr-sm border border-border"
              >
                <img
                  src={message.imageUrl}
                  alt={message.imageName ?? "Attached image"}
                  onError={() => setImageFailed(true)}
                  className="max-h-64 w-auto max-w-full object-contain"
                />
              </a>
            )}
            {message.content && (
              <div className="wrap-anywhere rounded-2xl rounded-tr-sm bg-primary px-4 py-2.5 text-[0.95rem] leading-7 text-primary-foreground">
                {message.content}
              </div>
            )}
          </div>
        ) : (
          <div className={cn(!done && "typing-caret")}>
            <Markdown content={shown} />
            {/* Shown as soon as it is known rather than waiting for `done`:
                it is a plain <img>, so unlike Mol* below there is no WebGL
                context to initialise, and the picture is the point of the
                answer - making the user read to the end first is worse. */}
            {showGenerated && (
              <a
                href={message.generatedImageUrl}
                target="_blank"
                rel="noreferrer"
                className="mt-3 block w-fit overflow-hidden rounded-2xl border border-border"
              >
                <img
                  src={message.generatedImageUrl}
                  alt="Generated scientific illustration"
                  onError={() => setGeneratedFailed(true)}
                  className="max-h-[28rem] w-auto max-w-full object-contain"
                />
              </a>
            )}
            {/* Shown once the text settles. Unlike Mol* there is nothing
                expensive to mount, but the scores are a footnote to the
                written answer and should not appear above it mid-type. */}
            {message.recognition && done && (
              <RecognitionPanel result={message.recognition} />
            )}
            {/* Held back until the text finishes typing: mounting Mol* mid-
                animation makes it initialise its WebGL context while the
                message above it is still reflowing on every tick. */}
            {message.proteinViewer && done && <ProteinViewer spec={message.proteinViewer} />}
            {/* Held back like the panels above: the frame pulls a ~700 KB
                document, and starting that download while the text is still
                reflowing on every tick just makes both feel slower. */}
            {/* Inline SVG, so there is nothing to fetch - but it is still
                held until the text settles, so the answer does not reflow
                around a chart appearing mid-sentence. */}
            {message.genomeChart && done && <GenomeChart spec={message.genomeChart} />}
            {/* Held until the text settles for the same reason as the panels
                above - but it matters more here: the answer's last line
                introduces this draft ("the abstract is below"), so revealing
                the draft first would put the text above its own introduction. */}
            {message.writingDraft && done && <WritingPanel spec={message.writingDraft} />}
            {message.biodiversityMap && done && (
              <BiodiversityMap spec={message.biodiversityMap} />
            )}
            {/* Held back like the panels above. This one also animates its
                progress bar on mount, and starting that while the text is
                still typing would run it against a moving layout. */}
            {message.reconstruction && done && (
              <ReconstructionPanel spec={message.reconstruction} />
            )}
          </div>
        )}
      </div>

      {isUser && <UserAvatar name={userName} className="mt-1" />}
    </div>
  );
}
