from io import BytesIO
from typing import Any

from pypdf import PdfReader


def load_uploaded_file(uploaded_file) -> list[dict[str, Any]]:
    """
    返回结构：
    [
        {
            "file_name": "xxx.pdf",
            "page": 1,
            "text": "..."
        }
    ]
    """

    file_name = uploaded_file.name
    lower_name = file_name.lower()

    if lower_name.endswith(".pdf"):
        return load_pdf(uploaded_file, file_name)

    if lower_name.endswith(".txt") or lower_name.endswith(".md"):
        raw = uploaded_file.getvalue()
        text = raw.decode("utf-8", errors="ignore")

        return [
            {
                "file_name": file_name,
                "page": None,
                "text": text,
            }
        ]

    raise ValueError("暂时只支持 PDF / TXT / Markdown 文件。")


def load_pdf(uploaded_file, file_name: str) -> list[dict[str, Any]]:
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    docs = []

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if text.strip():
            docs.append(
                {
                    "file_name": file_name,
                    "page": page_idx + 1,
                    "text": text,
                }
            )

    return docs