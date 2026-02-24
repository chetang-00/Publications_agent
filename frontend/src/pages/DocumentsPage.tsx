import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileText, Loader2, Trash2, UploadCloud } from "lucide-react";
import { useState, type DragEvent } from "react";
import { ApiError, api } from "../api/client";
import type { DocumentInfo, DocumentStatus } from "../api/types";
import { StatusBadge, type Tone } from "../components/StatusBadge";
import { formatBytes, relativeTime } from "../lib/format";

const ACCEPT = ".pdf,.docx,.txt,.md,.markdown";
const STATUS: Record<DocumentStatus, { label: string; tone: Tone }> = {
  processing: { label: "Processing", tone: "info" },
  ready: { label: "Ready", tone: "success" },
  failed: { label: "Failed", tone: "danger" },
};

export function pollInterval(docs: DocumentInfo[] | undefined): number | false {
  return docs?.some((d) => d.status === "processing") ? 1500 : false;
}

function describe(doc: DocumentInfo): string {
  return [
    formatBytes(doc.size_bytes),
    doc.page_count ? `${doc.page_count} page${doc.page_count === 1 ? "" : "s"}` : null,
    doc.chunk_count ? `${doc.chunk_count} passages` : null,
    relativeTime(doc.created_at),
  ]
    .filter(Boolean)
    .join(" · ");
}

export function DocumentsPage() {
  const queryClient = useQueryClient();
  const documents = useQuery({
    queryKey: ["documents"],
    queryFn: api.listDocuments,
    refetchInterval: (query) => pollInterval(query.state.data),
  });
  const [uploading, setUploading] = useState<string[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const remove = useMutation({
    mutationFn: api.deleteDocument,
    onSuccess: () => {
      setConfirming(null);
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
    onError: (err) => setErrors([err instanceof ApiError ? err.message : "Could not delete the document."]),
  });

  async function upload(files: File[]) {
    setErrors([]);
    for (const file of files) {
      setUploading((u) => [...u, file.name]);
      try {
        await api.uploadDocument(file);
      } catch (err) {
        setErrors((e) => [...e, `${file.name}: ${err instanceof ApiError ? err.message : "Upload failed."}`]);
      } finally {
        setUploading((u) => u.filter((name) => name !== file.name));
      }
    }
    await queryClient.invalidateQueries({ queryKey: ["documents"] });
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    void upload(Array.from(e.dataTransfer.files));
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-6 px-4 py-8">
        <header>
          <h1 className="text-xl font-semibold">Documents</h1>
          <p className="mt-1 text-sm text-zinc-500">
            Upload papers and notes for the assistant to search. PDF (with a text layer), DOCX, TXT and Markdown, up to 25 MB.
          </p>
        </header>

        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
            dragging
              ? "border-brand-500 bg-brand-50 dark:bg-brand-700/20"
              : "border-zinc-300 bg-white hover:border-brand-300 dark:border-zinc-700 dark:bg-zinc-900"
          }`}
        >
          <UploadCloud aria-hidden className="size-8 text-brand-600 dark:text-brand-300" />
          <span className="text-sm font-medium">Drop files here or click to choose</span>
          <span className="text-xs text-zinc-500">They are parsed, split into passages and indexed for search.</span>
          <input
            type="file"
            multiple
            accept={ACCEPT}
            aria-label="Upload documents"
            className="sr-only"
            onChange={(e) => {
              const files = Array.from(e.target.files ?? []);
              e.target.value = "";
              void upload(files);
            }}
          />
        </label>

        {uploading.length > 0 && (
          <p className="flex items-center gap-2 text-sm text-zinc-500">
            <Loader2 aria-hidden className="size-4 animate-spin" /> Uploading {uploading.join(", ")}…
          </p>
        )}
        {errors.length > 0 && (
          <div role="alert" className="space-y-1 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-200">
            {errors.map((message) => (
              <p key={message}>{message}</p>
            ))}
          </div>
        )}

        {documents.isPending && <Loader2 aria-label="Loading" className="size-5 animate-spin text-zinc-400" />}
        {documents.isError && <p className="text-sm text-rose-600">Could not load documents.</p>}
        {documents.data?.length === 0 && <p className="text-sm text-zinc-500">No documents yet.</p>}

        <ul className="divide-y divide-zinc-200 rounded-xl border border-zinc-200 bg-white dark:divide-zinc-800 dark:border-zinc-800 dark:bg-zinc-900">
          {documents.data?.map((doc) => {
            const status = STATUS[doc.status];
            return (
              <li key={doc.id} className="flex items-start gap-3 px-4 py-3">
                <FileText aria-hidden className="mt-0.5 size-5 shrink-0 text-zinc-400" />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="truncate text-sm font-medium">{doc.filename}</span>
                    <StatusBadge tone={status.tone} spinning={doc.status === "processing"}>
                      {status.label}
                    </StatusBadge>
                  </div>
                  <p className="mt-0.5 text-xs text-zinc-500">{describe(doc)}</p>
                  {doc.error && <p className="mt-1 text-xs text-rose-600 dark:text-rose-400">{doc.error}</p>}
                </div>
                {confirming === doc.id ? (
                  <div className="flex shrink-0 items-center gap-1">
                    <button
                      type="button"
                      aria-label={`Confirm delete ${doc.filename}`}
                      disabled={remove.isPending}
                      onClick={() => remove.mutate(doc.id)}
                      className="rounded-md bg-rose-600 px-2.5 py-1 text-xs font-semibold text-white hover:bg-rose-700 disabled:opacity-60"
                    >
                      Delete
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirming(null)}
                      className="rounded-md px-2.5 py-1 text-xs text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    aria-label={`Delete ${doc.filename}`}
                    disabled={doc.status === "processing"}
                    onClick={() => setConfirming(doc.id)}
                    className="shrink-0 rounded-md p-1.5 text-zinc-400 hover:bg-zinc-100 hover:text-rose-600 disabled:opacity-40 dark:hover:bg-zinc-800"
                  >
                    <Trash2 className="size-4" />
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
