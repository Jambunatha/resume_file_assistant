"""
fs_tools.py

Core file-system tools for the Resume File Assistant.

The assignment requires four tools:
1. read_file
2. list_files
3. write_file
4. search_in_file

PDF resumes are loaded using LangChain's PyPDFLoader.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

from langchain_community.document_loaders import PyPDFLoader


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()

    return {
        "name": path.name,
        "path": str(path.resolve()),
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_date": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using LangChain PyPDFLoader."""
    loader = PyPDFLoader(str(path))
    documents = loader.load()

    return "\n\n".join(
        document.page_content.strip()
        for document in documents
        if document.page_content.strip()
    )


def read_file(filepath: str) -> dict[str, Any]:
    """
    Read a PDF resume and return extracted text and metadata.
    """
    try:
        path = Path(filepath)

        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {filepath}",
            }

        if not path.is_file():
            return {
                "success": False,
                "error": f"Not a file: {filepath}",
            }

        if path.suffix.lower() != ".pdf":
            return {
                "success": False,
                "error": "Only PDF resume files are supported.",
            }

        content = _extract_pdf_text(path)

        return {
            "success": True,
            "content": content,
            "metadata": _file_metadata(path),
            "character_count": len(content),
            "word_count": len(content.split()),
        }

    except Exception as exc:
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def list_files(
    directory: str,
    extension: str | None = None,
) -> list[dict[str, Any]]:
    """
    List files in a directory.

    Optional extension example: ".pdf".
    """
    try:
        directory_path = Path(directory)

        if not directory_path.exists():
            return [{
                "success": False,
                "error": f"Directory not found: {directory}",
            }]

        if not directory_path.is_dir():
            return [{
                "success": False,
                "error": f"Not a directory: {directory}",
            }]

        normalized_extension = extension.lower() if extension else None

        if normalized_extension and not normalized_extension.startswith("."):
            normalized_extension = "." + normalized_extension

        results = []

        for path in sorted(
            directory_path.iterdir(),
            key=lambda item: item.name.lower(),
        ):
            if not path.is_file():
                continue

            if normalized_extension:
                if path.suffix.lower() != normalized_extension:
                    continue

            results.append(_file_metadata(path))

        return results

    except Exception as exc:
        return [{
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }]


def write_file(filepath: str, content: str) -> dict[str, Any]:
    """
    Write text content to a file and create parent directories if necessary.
    """
    try:
        path = Path(filepath)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "message": "File written successfully.",
            "metadata": _file_metadata(path),
        }

    except Exception as exc:
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def search_in_file(
    filepath: str,
    keyword: str,
) -> dict[str, Any]:
    """
    Search a PDF resume case-insensitively.

    Returns every match and surrounding text for context.
    """
    if not keyword.strip():
        return {
            "success": False,
            "error": "Keyword cannot be empty.",
        }

    result = read_file(filepath)

    if not result.get("success"):
        return result

    content = result["content"]
    context_size = 100
    matches = []

    for match in re.finditer(
        re.escape(keyword),
        content,
        flags=re.IGNORECASE,
    ):
        start = max(0, match.start() - context_size)
        end = min(len(content), match.end() + context_size)

        matches.append({
            "matched_text": match.group(0),
            "position": match.start(),
            "context": content[start:end].replace("\n", " ").strip(),
        })

    return {
        "success": True,
        "filepath": str(Path(filepath).resolve()),
        "keyword": keyword,
        "match_count": len(matches),
        "matches": matches,
    }
