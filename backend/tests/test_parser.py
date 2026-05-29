from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from backend.app.services.pdf.parser import parse_pdf


def make_sample_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, height - 72, "Sample Paper Title")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, height - 110, "1 Introduction")

    c.setFont("Helvetica", 11)
    c.drawString(72, height - 135, "This paper presents a simple parser for PDF papers.")
    c.drawString(72, height - 155, "It should chunk text recursively and keep metadata.")

    c.showPage()
    c.save()


def test_parse_pdf_creates_text_chunks_and_cache(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    cache_dir = tmp_path / "cache"
    make_sample_pdf(pdf_path)

    parsed = parse_pdf(pdf_path, cache_dir=cache_dir)

    assert parsed["doc_id"]
    assert parsed["title"]
    assert isinstance(parsed["raw_blocks"], list)
    assert isinstance(parsed["chunks"], list)
    assert parsed["chunks"]
    assert parsed["pages"]

    first_chunk = parsed["chunks"][0]
    assert first_chunk["type"] == "text"
    assert "content" in first_chunk
    assert "metadata" in first_chunk
    assert first_chunk["metadata"]["page_number"] == 1
    assert "chunk_index" in first_chunk["metadata"]
    assert cache_dir.joinpath(f'{parsed["doc_id"]}.json').exists()

    cached = parse_pdf(pdf_path, cache_dir=cache_dir)
    assert cached == parsed
