import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function Markdown({ content }: { content: string }) {
  return (
    <div className="space-y-4 text-[0.95rem] leading-7">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: (props) => <h1 className="mt-2 text-xl font-semibold" {...props} />,
          h2: (props) => <h2 className="mt-2 text-lg font-semibold" {...props} />,
          h3: (props) => <h3 className="mt-2 text-base font-semibold" {...props} />,
          p: (props) => <p className="text-foreground/90" {...props} />,
          ul: (props) => <ul className="list-disc space-y-1.5 pl-5" {...props} />,
          ol: (props) => <ol className="list-decimal space-y-1.5 pl-5" {...props} />,
          a: (props) => (
            <a className="text-primary underline underline-offset-4" target="_blank" rel="noreferrer" {...props} />
          ),
          blockquote: (props) => (
            <blockquote
              className="border-l-2 border-primary/60 bg-accent/40 px-4 py-2 text-sm text-muted-foreground"
              {...props}
            />
          ),
          code: ({ className, children, ...rest }) => {
            const isBlock = /language-/.test(className ?? "");
            if (!isBlock) {
              return (
                <code
                  className="rounded bg-muted px-1.5 py-0.5 font-[family-name:var(--font-mono-custom)] text-[0.85em]"
                  {...rest}
                >
                  {children}
                </code>
              );
            }
            return (
              <code
                className="block overflow-x-auto font-[family-name:var(--font-mono-custom)] text-[0.85rem] leading-6"
                {...rest}
              >
                {children}
              </code>
            );
          },
          pre: (props) => (
            <pre
              className="overflow-x-auto rounded-lg border border-border bg-card p-4 text-card-foreground"
              {...props}
            />
          ),
          table: (props) => (
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full border-collapse text-sm" {...props} />
            </div>
          ),
          th: (props) => (
            <th className="border-b border-border bg-muted/60 px-3 py-2 text-left font-medium" {...props} />
          ),
          td: (props) => <td className="border-b border-border/60 px-3 py-2 align-top" {...props} />,
          hr: () => <hr className="border-border" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}