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


def test_markdown_to_html_renders_table():
    md = "| Model | Params |\n|-------|--------|\n| GPT-2 | 117M |\n| GPT-3 | 175B |"
    html = markdown_to_html(md, title="Test")
    assert "<table" in html
    assert "GPT-2" in html
    assert "175B" in html


def test_markdown_to_html_mermaid_fallback(mocker):
    """When mmdc is unavailable, mermaid block falls back to a styled pre block."""
    mocker.patch("subprocess.run", side_effect=FileNotFoundError("mmdc not found"))
    md = "```mermaid\nflowchart TD\n    A --> B\n```"
    html = markdown_to_html(md, title="Test")
    assert "mermaid-fallback" in html
    assert "A --&gt; B" in html or "A --> B" in html


def test_markdown_to_html_mermaid_renders_png(mocker, tmp_path):
    """When mmdc succeeds, the PNG is embedded as a base64 data URI."""
    fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"

    def fake_run(cmd, **kwargs):
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_bytes(fake_png)
        result = mocker.MagicMock()
        result.returncode = 0
        return result

    mocker.patch("subprocess.run", side_effect=fake_run)
    md = "```mermaid\nflowchart TD\n    A --> B\n```"
    html = markdown_to_html(md, title="Test")
    assert 'class="diagram"' in html
    assert "data:image/png;base64," in html


@requires_weasyprint
def test_render_pdf_creates_file(tmp_path, mocker):
    mock_html_class = mocker.patch("weasyprint.HTML")
    mock_css_class = mocker.patch("weasyprint.CSS")
    output_path = str(tmp_path / "book.pdf")
    render_pdf("<html><body><p>Test</p></body></html>", output_path=output_path)
    mock_html_class.assert_called_once_with(string="<html><body><p>Test</p></body></html>")
