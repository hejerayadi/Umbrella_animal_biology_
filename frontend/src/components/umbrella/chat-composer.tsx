import { ArrowUp, Loader2, Mic, Paperclip, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ACCEPTED_IMAGE_TYPES, uploadImage, type UploadedImage } from "@/lib/orchestrator-client";
import { toast } from "sonner";

/** An image the user has attached and the backend has accepted. */
interface Attachment {
  /** The handle to send with the message. */
  uploaded: UploadedImage;
  /** Local object URL, for the thumbnail only - never sent anywhere. */
  previewUrl: string;
}

export function ChatComposer({
  onSend,
  disabled,
  focusKey,
}: {
  onSend: (value: string, image?: UploadedImage | null) => void;
  disabled?: boolean;
  focusKey?: string;
}) {
  const [value, setValue] = useState("");
  const [attachment, setAttachment] = useState<Attachment | null>(null);
  const [uploading, setUploading] = useState(false);
  const [draggingOver, setDraggingOver] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    ref.current?.focus();
  }, [focusKey, disabled]);

  // Object URLs are held by the browser until explicitly released. Revoking on
  // replace/unmount keeps a long conversation from leaking every image the
  // user ever attached.
  useEffect(() => {
    return () => {
      if (attachment) URL.revokeObjectURL(attachment.previewUrl);
    };
  }, [attachment]);

  const clearAttachment = () => {
    setAttachment((current) => {
      if (current) URL.revokeObjectURL(current.previewUrl);
      return null;
    });
    // Without this the same file cannot be re-picked: the input keeps its
    // value and fires no change event the second time.
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const attach = async (file: File) => {
    if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
      toast.error("Only JPEG, PNG and WebP images can be attached.");
      return;
    }

    setUploading(true);
    try {
      // Uploaded byte-for-byte, deliberately not re-encoded. The recognition
      // backend matches known images by the file's SHA-256, so re-compressing
      // here would silently stop those images being recognised. The size limit
      // is enforced server-side and its message is shown below.
      const uploaded = await uploadImage(file);
      setAttachment((current) => {
        if (current) URL.revokeObjectURL(current.previewUrl);
        return { uploaded, previewUrl: URL.createObjectURL(file) };
      });
    } catch (error: unknown) {
      // The backend's rejection messages are written for a person to act on.
      toast.error(error instanceof Error ? error.message : "The image could not be uploaded.");
    } finally {
      setUploading(false);
      ref.current?.focus();
    }
  };

  const submit = () => {
    const trimmed = value.trim();
    // An image on its own is a real message: the agent needs both, but the
    // instruction can be as short as the placeholder below.
    if ((!trimmed && !attachment) || disabled || uploading) return;

    onSend(trimmed || "What species is in this image?", attachment?.uploaded ?? null);
    setValue("");
    clearAttachment();
    ref.current?.focus();
  };

  const imageFrom = (items: DataTransferItemList | FileList | null): File | null => {
    if (!items) return null;
    const files = Array.from(items as ArrayLike<DataTransferItem | File>);
    for (const entry of files) {
      const file = entry instanceof File ? entry : entry.getAsFile();
      if (file && ACCEPTED_IMAGE_TYPES.includes(file.type)) return file;
    }
    return null;
  };

  return (
    <div
      className={`rounded-2xl border bg-card p-2 shadow-[var(--shadow-soft)] transition-colors ${
        draggingOver ? "border-primary border-dashed" : "border-border"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setDraggingOver(true);
      }}
      onDragLeave={() => setDraggingOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDraggingOver(false);
        const file = imageFrom(e.dataTransfer.files);
        if (file) void attach(file);
        else toast.error("Drop a JPEG, PNG or WebP image.");
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_IMAGE_TYPES.join(",")}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void attach(file);
        }}
      />

      {attachment && (
        <div className="mb-2 flex items-center gap-2 rounded-xl border border-border bg-muted/40 p-2">
          <img
            src={attachment.previewUrl}
            alt={attachment.uploaded.filename}
            className="h-12 w-12 rounded-lg object-cover"
          />
          <div className="min-w-0 flex-1">
            <p className="truncate text-[0.8rem] font-medium">{attachment.uploaded.filename}</p>
            <p className="text-[0.7rem] text-muted-foreground">
              {Math.max(1, Math.round(attachment.uploaded.size_bytes / 1024))} KB · will be sent for
              species recognition
            </p>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Remove attached image"
            onClick={clearAttachment}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      )}

      <Textarea
        ref={ref}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onPaste={(e) => {
          const file = imageFrom(e.clipboardData.items);
          if (file) {
            // Stop the image's filename being pasted in as text as well.
            e.preventDefault();
            void attach(file);
          }
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        rows={2}
        placeholder={
          attachment
            ? "Ask about this image… (or press Enter to just identify the species)"
            : "Ask Umbrella about genomes, traits, species, structures, or literature…"
        }
        className="max-h-48 min-h-[56px] resize-none border-0 bg-transparent px-2 py-2 text-[0.95rem] shadow-none focus-visible:ring-0 dark:bg-transparent"
      />
      <div className="flex items-center justify-between px-1 pt-1">
        <div className="flex items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Attach an image"
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
          >
            {uploading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Paperclip className="h-4 w-4" />
            )}
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
            {uploading ? "Uploading image…" : "Enter to send · attach, paste or drop an image"}
          </span>
        </div>
        <Button
          type="button"
          size="icon"
          onClick={submit}
          disabled={disabled || uploading || (!value.trim() && !attachment)}
          aria-label="Send message"
          className="size-9 rounded-full"
        >
          <ArrowUp className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
