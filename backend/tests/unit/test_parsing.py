import pytest

from app.rag.parsing import ParseError, UnsupportedFile, detect_kind, parse_document
from tests.doc_helpers import make_docx, make_encrypted_pdf, make_pdf

PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.mark.parametrize(
    ("filename", "head", "kind"),
    [
        ("paper.pdf", b"%PDF-1.7\n", "pdf"),
        ("Report.PDF", b"%PDF-1.4\n", "pdf"),
        ("notes.docx", b"PK\x03\x04rest", "docx"),
        ("readme.txt", b"hello", "txt"),
        ("readme.md", b"# Title", "md"),
        ("readme.markdown", b"# Title", "md"),
    ],
)
def test_detect_kind(filename, head, kind):
    assert detect_kind(filename, head) == kind


def test_unsupported_extension():
    with pytest.raises(UnsupportedFile) as exc:
        detect_kind("slides.pptx", b"PK\x03\x04")
    assert "PDF, DOCX, TXT or MD" in str(exc.value)


def test_upload_rejects_empty_file():
    with pytest.raises(UnsupportedFile) as exc:
        detect_kind("empty.pdf", b"")
    assert "empty" in str(exc.value)


def test_signature_mismatch_rejected():
    with pytest.raises(UnsupportedFile) as exc:
        detect_kind("photo.pdf", PNG_HEADER)
    assert "PDF" in str(exc.value)
    with pytest.raises(UnsupportedFile):
        detect_kind("photo.docx", PNG_HEADER)


def test_binary_text_file_rejected():
    with pytest.raises(UnsupportedFile):
        detect_kind("data.txt", PNG_HEADER)


def test_pdf_pages_keep_their_numbers(tmp_path):
    path = make_pdf(
        tmp_path / "a.pdf", ["Methods: we enrolled 120 patients.", "", "Results: survival improved."]
    )
    doc = parse_document(path, "pdf")
    assert doc.page_count == 3
    assert [p.page for p in doc.pages] == [1, 3]
    assert "120 patients" in doc.pages[0].text
    assert "survival improved" in doc.pages[1].text


def test_pdf_without_text_layer_fails_with_reason(tmp_path):
    path = make_pdf(tmp_path / "scan.pdf", ["", ""])
    with pytest.raises(ParseError) as exc:
        parse_document(path, "pdf")
    assert "No extractable text" in str(exc.value)


@pytest.mark.parametrize("algorithm", ["RC4-128", "AES-128", "AES-256"])
def test_encrypted_pdf_fails_with_reason(tmp_path, algorithm):
    path = make_encrypted_pdf(tmp_path / "locked.pdf", "Confidential results", algorithm=algorithm)
    with pytest.raises(ParseError) as exc:
        parse_document(path, "pdf")
    assert "password" in str(exc.value)


@pytest.mark.parametrize("algorithm", ["RC4-128", "AES-128", "AES-256"])
def test_pdf_with_only_owner_restrictions_is_read(tmp_path, algorithm):
    # Publisher PDFs often restrict copying/printing but open without a password.
    path = make_encrypted_pdf(
        tmp_path / "restricted.pdf", "Results improved survival.", password="", algorithm=algorithm
    )
    doc = parse_document(path, "pdf")
    assert "survival" in doc.pages[0].text


def test_corrupt_pdf_fails_with_reason(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\nthis is not really a pdf")
    with pytest.raises(ParseError) as exc:
        parse_document(path, "pdf")
    assert "Could not read" in str(exc.value)


def test_docx_paragraphs_and_tables(tmp_path):
    path = make_docx(
        tmp_path / "a.docx",
        ["Study design overview.", "Participants were adults."],
        table=[["Group", "N"], ["Treatment", "60"]],
    )
    doc = parse_document(path, "docx")
    assert doc.page_count == 1
    assert doc.pages[0].page is None
    text = doc.pages[0].text
    assert "Study design overview." in text
    assert "Treatment" in text and "60" in text


def test_corrupt_docx_fails_with_reason(tmp_path):
    path = tmp_path / "bad.docx"
    path.write_bytes(b"PK\x03\x04garbage")
    with pytest.raises(ParseError):
        parse_document(path, "docx")


def test_text_file_with_bom_and_crlf(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes("﻿Line one\r\nLine two\r\n\r\n\r\n\r\nLine three".encode())
    doc = parse_document(path, "txt")
    assert doc.pages[0].text == "Line one\nLine two\n\nLine three"


def test_non_utf8_text_is_tolerated(tmp_path):
    path = tmp_path / "latin.txt"
    path.write_bytes("Caf\xe9 au lait and enough other words".encode("latin-1"))
    assert "au lait" in parse_document(path, "txt").pages[0].text


def test_whitespace_only_text_file_fails(tmp_path):
    path = tmp_path / "blank.md"
    path.write_text("   \n\n  ")
    with pytest.raises(ParseError):
        parse_document(path, "md")
