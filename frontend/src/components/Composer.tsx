import { ArrowUp } from "lucide-react";
import { useLayoutEffect, useRef, useState } from "react";

export function Composer({
  disabled,
  onSend,
  hint,
  placeholder = "Ask about publications, your documents, or anything else…",
}: {
  disabled: boolean;
  onSend: (text: string) => void;
  hint?: string;
  placeholder?: string;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with the text. While empty, leave the height to rows={1}: measuring the placeholder
  // before the layout settles can yield a very tall box.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "";
    if (text) el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [text]);

  function submit() {
    const value = text.trim();
    if (!value || disabled) return;
    onSend(value);
    setText("");
  }

  return (
    <div className="mx-auto w-full max-w-3xl">
      {hint && <p className="mb-1.5 px-1 text-xs text-amber-700 dark:text-amber-400">{hint}</p>}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
        className="flex items-end gap-2 rounded-2xl border border-zinc-300 bg-white p-2 shadow-sm focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-100 dark:border-zinc-700 dark:bg-zinc-900 dark:focus-within:ring-brand-700/30"
      >
        <textarea
          ref={ref}
          aria-label="Message"
          rows={1}
          value={text}
          maxLength={8000}
          placeholder={placeholder}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          className="max-h-56 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] outline-none placeholder:text-zinc-400"
        />
        <button
          type="submit"
          aria-label="Send"
          disabled={disabled || !text.trim()}
          className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-brand-600 text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-zinc-300 dark:disabled:bg-zinc-700"
        >
          <ArrowUp className="size-5" />
        </button>
      </form>
      <p className="mt-1.5 px-1 text-center text-[11px] text-zinc-400">
        Enter to send · Shift+Enter for a new line · Answers cite their sources; check important details.
      </p>
    </div>
  );
}
