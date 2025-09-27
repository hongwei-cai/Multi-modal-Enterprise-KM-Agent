import os

import pytest

from src.rag.document_parser import parse_pdf

# Define the path to test PDFs
TEST_PDFS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")


def test_parse_valid_pdf():
    # Use an actual test PDF from the test data directory
    pdf_path = os.path.join(TEST_PDFS_DIR, "test_document_simple.pdf")
    if os.path.exists(pdf_path):
        result = parse_pdf(pdf_path)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Simple PDF for Unit Testing" in result
    else:
        pytest.skip("Test PDF not found")


def test_parse_invalid_pdf():
    with pytest.raises(FileNotFoundError):
        parse_pdf("nonexistent.pdf")


def test_parse_non_pdf_file():
    # Test with a non-PDF file
    txt_path = os.path.join(TEST_PDFS_DIR, "test_document_edge_case_messy.text-style")
    if os.path.exists(txt_path):
        with pytest.raises(ValueError):
            parse_pdf(txt_path)
    else:
        pytest.skip("Test file not found")


def test_pdf_parsing_multi_page():
    pdf_path = os.path.join(TEST_PDFS_DIR, "test_document_multi_page.pdf")
    if os.path.exists(pdf_path):
        result = parse_pdf(pdf_path)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Multi-Page PDF - Page 1" in result
        assert "Multi-Page PDF - Page 2" in result
        assert "Multi-Page PDF - Page 3" in result
    else:
        pytest.skip("Test PDF (multi-page) not found")


def test_pdf_parsing_with_table():
    pdf_path = os.path.join(TEST_PDFS_DIR, "test_document_table.pdf")
    if os.path.exists(pdf_path):
        result = parse_pdf(pdf_path)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "PDF with Simulated Table Layout" in result
        assert "Column 1" in result
    else:
        pytest.skip("Test PDF (with table) not found")


def test_pdf_parsing_empty():
    pdf_path = os.path.join(TEST_PDFS_DIR, "test_document_empty.pdf")
    if os.path.exists(pdf_path):
        result = parse_pdf(pdf_path)
        assert isinstance(result, str)
        assert len(result) == 0
    else:
        pytest.skip("Test PDF (empty) not found")
