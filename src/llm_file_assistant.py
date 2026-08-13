"""
llm_file_assistant.py

Cohere-powered Resume File Assistant.

The LLM can call the four required file tools:
    1. read_file()
    2. list_files()
    3. write_file()
    4. search_in_file()

Technology:
    - Cohere ClientV2
    - Cohere Command A
    - LangChain PyPDFLoader
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cohere import ClientV2

from fs_tools import (
    read_file,
    list_files,
    write_file,
    search_in_file,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESUME_DIRECTORY = PROJECT_ROOT / "documents" / "resumes"
OUTPUT_DIRECTORY = PROJECT_ROOT / "output"


# ============================================================
# COHERE CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "COHERE_MODEL",
    "command-a-03-2025",
)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise RuntimeError(
        "COHERE_API_KEY is not set.\n"
        "Please set your Cohere API key before running the application."
    )

cohere_client = ClientV2(
    api_key=COHERE_API_KEY
)


# ============================================================
# TOOL DEFINITIONS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a PDF resume using LangChain PyPDFLoader "
                "and return the extracted text and file metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": (
                            "Path to the PDF resume that should be read."
                        ),
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files in a directory. "
                "Use '.pdf' as the extension when listing resume PDFs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": (
                            "Directory containing the resume files."
                        ),
                    },
                    "extension": {
                        "type": ["string", "null"],
                        "description": (
                            "Optional file extension such as '.pdf'."
                        ),
                    },
                },
                "required": [
                    "directory",
                    "extension",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file. "
                "Create parent directories if they do not exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": (
                            "Destination path of the text file."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Text content that should be written."
                        ),
                    },
                },
                "required": [
                    "filepath",
                    "content",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": (
                "Search a PDF resume for a keyword or phrase "
                "case-insensitively and return matching text with context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": (
                            "Path to the PDF resume."
                        ),
                    },
                    "keyword": {
                        "type": "string",
                        "description": (
                            "Keyword or phrase to search for."
                        ),
                    },
                },
                "required": [
                    "filepath",
                    "keyword",
                ],
            },
        },
    },
]


# ============================================================
# PYTHON FUNCTION MAP
# ============================================================

FUNCTIONS = {
    "read_file": read_file,
    "list_files": list_files,
    "write_file": write_file,
    "search_in_file": search_in_file,
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are a professional Resume File Assistant.

The resume directory is:

{RESUME_DIRECTORY}

The output directory is:

{OUTPUT_DIRECTORY}

You have access to four file tools:

1. read_file
2. list_files
3. write_file
4. search_in_file

Follow these rules:

- Resume source files are PDF files.
- Never invent information that is not present in a resume.
- Use the available tools whenever the user asks to inspect, search,
  list, or create files.
- If the user asks to list resumes, use list_files.
- If the user asks to read a resume, use read_file.
- If the user asks to find resumes mentioning a skill or keyword,
  use list_files first and then search_in_file for the relevant PDF files.
- If the user asks to create a resume summary, first use read_file,
  then create a concise professional summary from the extracted content,
  and finally use write_file to save the summary.
- Do not claim that a file was created unless write_file succeeds.
- Give a clear final answer describing what was done.
"""


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """
    Execute one of the approved Python file tools.
    """

    function = FUNCTIONS.get(tool_name)

    if function is None:
        return {
            "success": False,
            "error": f"Unknown tool requested: {tool_name}",
        }

    try:
        return function(**arguments)

    except Exception as exc:
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ============================================================
# RESPONSE TEXT EXTRACTION
# ============================================================

def get_response_text(response: Any) -> str:
    """
    Extract the actual text response from a Cohere V2 response.

    Cohere responses can contain multiple content blocks, including
    thinking blocks and text blocks.

    Therefore, do not assume:

        response.message.content[0].text

    is always valid.
    """

    content = response.message.content or []

    for content_item in content:

        content_type = getattr(
            content_item,
            "type",
            None,
        )

        if content_type == "text":

            text = getattr(
                content_item,
                "text",
                None,
            )

            if text:
                return text

    return (
        "The model completed the request but did not "
        "return a text response."
    )


# ============================================================
# LLM TOOL-CALLING LOOP
# ============================================================

def ask_llm(user_query: str) -> str:
    """
    Send the user's request to Cohere.

    The model can request one or more Python tools.
    Tool results are sent back to Cohere until the model
    produces a final text response.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    while True:

        # ----------------------------------------------------
        # Send request to Cohere
        # ----------------------------------------------------

        response = cohere_client.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            temperature=0.2,
        )

        # ----------------------------------------------------
        # Check whether Cohere requested tools
        # ----------------------------------------------------

        tool_calls = response.message.tool_calls or []

        # ----------------------------------------------------
        # No tool calls = final response
        # ----------------------------------------------------

        if not tool_calls:
            return get_response_text(response)

        # ----------------------------------------------------
        # Add Cohere's assistant message containing tool calls
        # ----------------------------------------------------

        messages.append(response.message)

        # ----------------------------------------------------
        # Execute every requested tool
        # ----------------------------------------------------

        for tool_call in tool_calls:

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError as exc:

                result = {
                    "success": False,
                    "error": (
                        "Invalid tool arguments returned by the model: "
                        f"{exc}"
                    ),
                }

            else:

                result = execute_tool(
                    tool_call.function.name,
                    arguments,
                )

            # ------------------------------------------------
            # Convert Python result to JSON
            # ------------------------------------------------

            result_json = json.dumps(
                result,
                ensure_ascii=False,
            )

            # ------------------------------------------------
            # Send tool result back to Cohere
            # ------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": [
                        {
                            "type": "document",
                            "document": {
                                "data": result_json,
                            },
                        }
                    ],
                }
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Command-line entry point.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Cohere Resume File Assistant"
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="Natural-language request.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Optional Cohere model override.",
    )

    args = parser.parse_args()

    global MODEL_NAME

    if args.model:
        MODEL_NAME = args.model

    query = " ".join(args.query)

    print()
    print(f"Using Cohere model: {MODEL_NAME}")
    print()

    try:

        answer = ask_llm(query)

        print(answer)

    except Exception as exc:

        print()
        print("An error occurred:")
        print(f"{type(exc).__name__}: {exc}")


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()