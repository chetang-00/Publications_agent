"""End-to-end smoke test and agent evaluation against the running stack.

Drives the deployed app over HTTP exactly as the browser does (real Portkey gateway, real Qdrant,
real seeded data): checks health and readiness, uploads a document, then asks a set of golden
questions and grades each answer on the tools the agent chose, the facts in the answer and its
citations. Expected facts are computed from the seed CSV, not hard-coded.

    make smoke                     # or: cd backend && uv run python ../scripts/smoke.py
    ... --base-url http://127.0.0.1:8080 --keep --only count_2021,document
"""

import argparse
import json
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agent.events import agent_event_adapter  # noqa: E402
from app.seed.publications import read_csv  # noqa: E402
from app.text import normalize  # noqa: E402

COUNT_TOOLS = {"publication_stats", "filter_publications", "run_readonly_sql"}
PASS_THRESHOLD = 10
DOC_TEXT = (
    "Study protocol notes. We enrolled 120 adult patients across three clinical sites. "
    "The primary outcome was 12-month survival; the secondary outcome was quality of life."
)


def make_protocol_pdf(path: Path) -> Path:
    """A PDF that is AES-encrypted with owner restrictions only, like many publisher copies."""
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas

    plain = path.with_suffix(".plain.pdf")
    pdf = canvas.Canvas(str(plain))
    y = 800
    for line in (DOC_TEXT[i : i + 90] for i in range(0, len(DOC_TEXT), 90)):
        pdf.drawString(40, y, line)
        y -= 14
    pdf.showPage()
    pdf.save()
    writer = PdfWriter()
    for page in PdfReader(plain).pages:
        writer.add_page(page)
    writer.encrypt(user_password="", owner_password="smoke-owner", algorithm="AES-128")
    with path.open("wb") as handle:
        writer.write(handle)
    return path


@dataclass
class Case:
    id: str
    question: str
    tools_any: set[str] | None  # at least one of these must be called; empty set = no tools allowed
    facts: list[list[str]] = field(default_factory=list)  # each inner list: any one must appear
    min_citations: int = 0
    cited_titles_any: set[str] | None = None  # normalised titles; one citation must match
    attach_document: bool = False
    reject_approval: bool = False


@dataclass
class Outcome:
    case: Case
    tools: list[str] = field(default_factory=list)
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    approval_seen: bool = False
    error: str | None = None
    problems: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.problems


def ground_truth(csv_path: Path) -> dict:
    rows = read_csv(csv_path).rows
    years = Counter(r.year for r in rows if r.year)
    top_year, top_year_count = max(years.items(), key=lambda kv: (kv[1], kv[0]))
    journals = Counter(r.source_title for r in rows if r.source_title)
    authors = Counter(a for r in rows for a in r.authors)
    top_author, top_author_count = authors.most_common(1)[0]
    most_cited = max(rows, key=lambda r: r.cited_by or -1)
    return {
        "papers_2021": years[2021],
        "top_year": top_year,
        "top_year_count": top_year_count,
        "unlabelled": sum(1 for r in rows if r.cluster_label is None),
        "crispr": sum(1 for r in rows if r.cluster_label == "CRISPR Functional Genomics"),
        "top_journal": journals.most_common(1)[0][0],
        "top_author_surname": top_author.split()[0].rstrip(","),
        "top_author_count": top_author_count,
        "max_cited": most_cited.cited_by,
        "ranganath_2024_titles": {
            normalize(r.title)
            for r in rows
            if r.year == 2024 and any(normalize(a) == "ranganath r" for a in r.authors)
        },
    }


def build_cases(truth: dict) -> list[Case]:
    return [
        Case(
            "count_2021",
            "How many papers in the catalogue were published in 2021?",
            COUNT_TOOLS,
            [[str(truth["papers_2021"])]],
        ),
        Case(
            "top_year",
            "Which publication year has the most papers, and how many papers is that?",
            COUNT_TOOLS,
            [[str(truth["top_year"])], [str(truth["top_year_count"])]],
        ),
        Case(
            "unlabelled", "How many papers have no cluster label?", COUNT_TOOLS, [[str(truth["unlabelled"])]]
        ),
        Case(
            "crispr_cluster",
            "How many papers are in the 'CRISPR Functional Genomics' cluster?",
            COUNT_TOOLS,
            [[str(truth["crispr"])]],
        ),
        Case(
            "top_journal",
            "Which journal or conference has published the most papers in this catalogue?",
            COUNT_TOOLS,
            [[truth["top_journal"], truth["top_journal"][:30]]],
        ),
        Case(
            "most_cited",
            "What is the most cited paper in the catalogue? Give its title and citation count.",
            COUNT_TOOLS,
            [[str(truth["max_cited"])]],
            min_citations=1,
        ),
        Case(
            "top_author",
            "Who is the most frequent author in the catalogue, and how many papers do they have?",
            COUNT_TOOLS,
            [[truth["top_author_surname"]], [str(truth["top_author_count"])]],
        ),
        Case(
            "author_year",
            "List the papers Rajesh Ranganath published in 2024.",
            {"resolve_author", "filter_publications", "search_publications", "run_readonly_sql"},
            min_citations=1,
            cited_titles_any=truth["ranganath_2024_titles"],
        ),
        Case(
            "semantic",
            "Find papers about denoising score matching for diffusion models.",
            {"search_publications"},
            min_citations=1,
        ),
        Case(
            "document",
            "According to my uploaded protocol notes, how many patients were enrolled?",
            {"search_documents"},
            [["120"]],
            min_citations=1,
            attach_document=True,
        ),
        Case(
            "general",
            "In one sentence, what is a randomized controlled trial? Do not look anything up.",
            set(),
        ),
        Case(
            "approval",
            "Change the cluster label of publication 1 to 'Smoke Test Label'. The reason is: smoke test.",
            {"update_cluster_label"},
            reject_approval=True,
        ),
    ]


def stream(client: httpx.Client, path: str, body: dict, outcome: Outcome) -> dict | None:
    """POST and consume the SSE stream; returns the approval event if the run paused for one."""
    approval = None
    with client.stream("POST", path, json=body, timeout=httpx.Timeout(10, read=300)) as response:
        if response.status_code != 200:
            outcome.error = f"HTTP {response.status_code}: {response.read().decode()[:200]}"
            return None
        buffer = ""
        for chunk in response.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                data = "\n".join(line[6:] for line in block.split("\n") if line.startswith("data: "))
                if not data:
                    continue
                event = agent_event_adapter.validate_python(json.loads(data))
                if event.type == "tool_call_started":
                    outcome.tools.append(event.name)
                elif event.type == "approval_required":
                    outcome.approval_seen = True
                    approval = event.model_dump()
                elif event.type == "message_completed":
                    outcome.answer = event.content
                    outcome.citations = [c.model_dump() for c in event.citations]
                elif event.type == "error":
                    outcome.error = f"{event.code}: {event.message}"
    return approval


def grade(outcome: Outcome, client: httpx.Client, label_before: str | None) -> None:
    case = outcome.case
    problems = outcome.problems
    if outcome.error:
        problems.append(f"run error: {outcome.error}")
    used = set(outcome.tools)
    if case.tools_any == set() and used:
        problems.append(f"expected no tools, used {sorted(used)}")
    elif case.tools_any and not used & case.tools_any:
        problems.append(f"expected one of {sorted(case.tools_any)}, used {sorted(used) or 'none'}")
    answer = outcome.answer.replace(",", "").lower()
    for options in case.facts:
        if not any(option.replace(",", "").lower() in answer for option in options):
            problems.append(f"answer lacks {options[0]!r}")
    if len(outcome.citations) < case.min_citations:
        problems.append(f"expected ≥{case.min_citations} citation(s), got {len(outcome.citations)}")
    if case.cited_titles_any is not None:
        cited = {normalize(c.get("title") or "") for c in outcome.citations}
        if not cited & case.cited_titles_any:
            problems.append("no citation points to an expected paper")
    if case.tools_any == set() and not outcome.answer:
        problems.append("empty answer")
    if case.reject_approval:
        if not outcome.approval_seen:
            problems.append("write tool did not pause for approval")
        label_after = client.get("/api/publications/1").json()["cluster_label"]
        if label_after != label_before:
            problems.append(f"label changed despite rejection: {label_before!r} -> {label_after!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--csv", type=Path, default=ROOT / "data/seed/papers_with_cluster_labels.csv")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--keep", action="store_true", help="keep the conversations and document created")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=30)
    print(f"→ {args.base_url}")
    try:
        health = client.get("/api/health")
    except httpx.HTTPError as exc:
        print(f"✗ cannot reach the app: {exc}. Start it with `make up`.")
        return 1
    print(f"✓ health {health.status_code}")
    ready = client.get("/api/ready")
    for name, check in ready.json()["checks"].items():
        print(f"  {'✓' if check['ok'] else '✗'} {name}: {check['detail']}")
    if ready.status_code != 200:
        print("✗ not ready; fix the failing checks above (usually `make seed`).")
        return 1

    truth = ground_truth(args.csv)

    pdf_path = make_protocol_pdf(Path(tempfile.mkdtemp()) / "protocol-notes.pdf")
    with pdf_path.open("rb") as data:
        doc = client.post(
            "/api/documents", files={"file": ("protocol-notes.pdf", data, "application/pdf")}
        ).json()
    deadline = time.time() + 120
    while doc["status"] == "processing" and time.time() < deadline:
        time.sleep(1)
        doc = client.get(f"/api/documents/{doc['id']}").json()
    print(
        f"{'✓' if doc['status'] == 'ready' else '✗'} document upload: {doc['status']} {doc.get('error') or ''}"
    )

    cases = build_cases(truth)
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.id in wanted]

    outcomes: list[Outcome] = []
    created: list[str] = []
    for case in cases:
        outcome = Outcome(case)
        started = time.time()
        conversation = client.post("/api/conversations", json={"title": f"smoke: {case.id}"}).json()["id"]
        created.append(conversation)
        if case.attach_document and doc["status"] == "ready":
            client.post(f"/api/conversations/{conversation}/documents", json={"document_ids": [doc["id"]]})
        label_before = (
            client.get("/api/publications/1").json()["cluster_label"] if case.reject_approval else None
        )
        approval = stream(
            client, f"/api/conversations/{conversation}/messages", {"content": case.question}, outcome
        )
        if approval and case.reject_approval:
            stream(
                client,
                f"/api/runs/{approval['run_id']}/approvals",
                {
                    "tool_call_id": approval["tool_call_id"],
                    "approved": False,
                    "note": "Rejected by the smoke test.",
                },
                outcome,
            )
        outcome.seconds = time.time() - started
        grade(outcome, client, label_before)
        outcomes.append(outcome)
        mark = "✓" if outcome.passed else "✗"
        print(f"{mark} {case.id:<15} {outcome.seconds:5.1f}s  tools={','.join(outcome.tools) or '-'}")
        for problem in outcome.problems:
            print(f"    - {problem}")
        if not outcome.passed and outcome.answer:
            one_line = re.sub(r"\s+", " ", outcome.answer)
            print(f"    answer: {one_line[:300]}")

    if not args.keep:
        for conversation in created:
            client.delete(f"/api/conversations/{conversation}")
        client.delete(f"/api/documents/{doc['id']}")

    passed = sum(o.passed for o in outcomes)
    target = min(PASS_THRESHOLD, len(outcomes)) if not args.only else len(outcomes)
    print(f"\n{passed}/{len(outcomes)} cases passed (target ≥ {target})")
    return 0 if passed >= target else 1


if __name__ == "__main__":
    sys.exit(main())
