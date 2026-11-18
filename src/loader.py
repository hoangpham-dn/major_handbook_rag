'''
Functions to preprocess and load pdf files 
'''
import logging
import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf_files(directory: str) -> list[Document]:
    docs = []

    for pdf in sorted(Path(directory).rglob("*.pdf")):
        logger.info("Loading: %s", pdf.name)
        file_path = str(pdf.resolve())

        try:
            with pymupdf.open(file_path) as pdf_file:
                for page_num, page in enumerate(pdf_file):
                    base_metadata = {
                        "source": pdf.name,
                        "file_path": file_path,
                        "page": page_num + 1,
                    }

                    # 1. Tables as their own documents
                    tables = page.find_tables().tables
                    table_bboxes = []

                    for table_idx, table in enumerate(tables):
                        table_bboxes.append(table.bbox)
                        table_content = table.to_markdown()

                        if not table_content.strip():
                            continue

                        docs.append(
                            Document(
                                page_content=table_content,
                                metadata={
                                    **base_metadata,
                                    "element_type": "table",
                                    "table_id": f"table_{table_idx + 1}",
                                },
                            )
                        )

                    # 2. Text blocks outside tables
                    page_text = []

                    for block in page.get_text("blocks", sort=True):
                        x0, y0, x1, y1, text, _, block_type = block[:7]

                        if block_type != 0 or not text.strip():
                            continue

                        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                        if any(
                            tx0 <= cx <= tx1 and ty0 <= cy <= ty1
                            for tx0, ty0, tx1, ty1 in table_bboxes
                        ):
                            continue

                        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
                        page_text.append(" ".join(text.split()))

                    content = "\n\n".join(page_text)

                    if content.strip():
                        docs.append(
                            Document(
                                page_content=content,
                                metadata={**base_metadata, "element_type": "text"},
                            )
                        )
                    elif not tables:
                        logger.warning("%s p.%d: no text", pdf.name, page_num + 1)

        except Exception:
            logger.exception("Failed to process PDF: %s", pdf.name)

    return docs


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    DATA_DIR = BASE_DIR.parent / "data" / "major_handbooks"

    documents = load_pdf_files(str(DATA_DIR))

    print(f"Loaded {len(documents)} pages.")
