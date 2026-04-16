import pytest
from pathlib import Path
from pipeline.pdf_renderer import markdown_to_html, render_pdf

# Skip WeasyPrint-dependent tests if system libraries (pango, cairo) are not available
try:
    import weasyprint  # noqa: F401
    weasyprint_available = True
except OSError:
    weasyprint_available = False

requires_weasyprint = pytest.mark.skipif(
    not weasyprint_available,
    reason="WeasyPrint system libraries (pango/cairo) not available",
)


def test_markdown_to_html_converts_headings():
    md = "## Introduction\n\nSome text here."
    html = markdown_to_html(md, title="Test Book")
    assert "<h2" in html
    assert "Some text here." in html
    assert "Test Book" in html


def test_markdown_to_html_converts_code_blocks():
    md = "```python\ndef hello():\n    pass\n```"
    html = markdown_to_html(md, title="Test Book")
    assert "<code" in html or "<pre" in html


def test_markdown_to_html_wraps_in_html_document():
    html = markdown_to_html("Hello world", title="My Book")
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "My Book" in html


@requires_weasyprint
def test_render_pdf_creates_file(tmp_path, mocker):
    mock_html_class = mocker.patch("weasyprint.HTML")
    mock_css_class = mocker.patch("weasyprint.CSS")
    output_path = str(tmp_path / "book.pdf")
    render_pdf("<html><body><p>Test</p></body></html>", output_path=output_path)
    mock_html_class.assert_called_once_with(string="<html><body><p>Test</p></body></html>")
