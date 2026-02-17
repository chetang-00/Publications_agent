import itertools
import re

from app.rag.chunking import chunk_pages, estimate_tokens
from app.rag.parsing import ParsedPage


def sentences(n: int) -> str:
    return " ".join(f"Sentence number {i} talks about topic {i % 7} in some detail." for i in range(n))


def test_short_page_is_one_chunk():
    chunks = chunk_pages([ParsedPage(page=4, text="Short text.")])
    assert len(chunks) == 1
    assert (chunks[0].index, chunks[0].page, chunks[0].text) == (0, 4, "Short text.")


def test_long_text_chunks_respect_size_and_overlap():
    text = sentences(400)
    chunks = chunk_pages([ParsedPage(page=1, text=text)], max_chars=800, overlap_chars=120)
    assert len(chunks) > 5
    assert all(len(c.text) <= 800 for c in chunks)
    for previous, current in itertools.pairwise(chunks):
        tail_words = previous.text.split()[-3:]
        assert " ".join(tail_words) in current.text  # consecutive chunks overlap
    covered = " ".join(c.text for c in chunks)
    for i in range(400):
        assert f"Sentence number {i} " in covered


def test_chunks_never_cross_pages_and_are_numbered():
    pages = [ParsedPage(page=1, text=sentences(60)), ParsedPage(page=2, text="Page two only.")]
    chunks = chunk_pages(pages, max_chars=800, overlap_chars=100)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].page == 2
    assert chunks[-1].text == "Page two only."
    assert all("Page two" not in c.text for c in chunks if c.page == 1)


def test_paragraph_breaks_are_preferred():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 100 for i in range(6))
    chunks = chunk_pages([ParsedPage(page=1, text=text)], max_chars=1200, overlap_chars=0)
    for chunk in chunks:
        assert re.match(r"Paragraph \d+\.", chunk.text)


def test_unbreakable_text_is_hard_split():
    chunks = chunk_pages([ParsedPage(page=None, text="x" * 2500)], max_chars=1000, overlap_chars=100)
    assert all(len(c.text) <= 1000 for c in chunks)
    assert sum(len(c.text) for c in chunks) >= 2500


def test_token_estimate():
    assert estimate_tokens("abcd" * 100) == 100
    assert estimate_tokens("a") == 1
    chunk = chunk_pages([ParsedPage(page=1, text="abcd" * 10)])[0]
    assert chunk.token_count == 10
