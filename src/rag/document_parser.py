"""
Document parsing utilities for the RAG system.
"""

import logging
from pathlib import Path

try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
except ImportError as exc:
    raise ImportError(
        "pypdf is required for PDF parsing. Install it with: pip install pypdf"
    ) from exc

# Configure logging
logger = logging.getLogger(__name__)


class PDFParser:
    """
    PDF document parser using pypdf for text extraction.
    """

    def __init__(self):
        """Initialize the PDF parser."""
        self.logger = logging.getLogger(__name__)

    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content as a string

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            PdfReadError: If the PDF file is corrupted or unreadable
            ValueError: If the file is not a PDF
        """
        file_path_obj = Path(file_path)

        # Validate file exists
        if not file_path_obj.exists():
            error_msg = f"PDF file not found: {file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Validate file is a PDF
        if file_path_obj.suffix.lower() != ".pdf":
            error_msg = f"File is not a PDF: {file_path}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            self.logger.info("Starting PDF text extraction from: %s", file_path)

            # Create PDF reader
            with open(file_path, "rb") as file:
                pdf_reader = PdfReader(file)

                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    self.logger.warning(
                        "PDF is encrypted: %s. Attempting to decrypt with \
                        empty password.",
                        file_path,
                    )
                    try:
                        pdf_reader.decrypt("")
                    except Exception as e:
                        error_msg = (
                            f"Failed to decrypt PDF: {file_path}. Error: {str(e)}"
                        )
                        self.logger.error(error_msg)
                        raise PdfReadError(error_msg) from e

                # Extract text from all pages
                text_content = []
                num_pages = len(pdf_reader.pages)
                self.logger.info("PDF has %d pages", num_pages)

                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text_content.append(page_text)
                        self.logger.debug(
                            "Extracted text from page %d/%d", page_num + 1, num_pages
                        )
                    except ValueError as e:
                        self.logger.warning(
                            "Failed to extract text from page %d: %s",
                            page_num + 1,
                            str(e),
                        )
                        continue

                # Join all page texts
                full_text = "\n\n".join(text_content)

                self.logger.info(
                    "Successfully extracted %d characters from %d pages",
                    len(full_text),
                    len(text_content),
                )
                return full_text

        except PdfReadError as e:
            error_msg = f"PDF read error for {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise PdfReadError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error parsing PDF {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e


def parse_pdf(file_path: str) -> str:
    """
    Convenience function to parse a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    parser = PDFParser()
    return parser.extract_text(file_path)
