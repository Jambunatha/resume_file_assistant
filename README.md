# Resume File Assistant

A simple command-line Resume File Assistant.

The project uses **Cohere Command A for LLM tool calling** and
**LangChain PyPDFLoader for PDF resume extraction**.

## Technology Stack

- Python 3.10+
- Cohere ClientV2
- Cohere Command A (`command-a-03-2025`)
- LangChain `PyPDFLoader`
- PyPDF
- ReportLab (only used to create the included dummy PDF resumes)

## Project Structure

```text
resume_file_assistant/
│
├── src/
│   ├── fs_tools.py
│   └── llm_file_assistant.py
│
├── documents/
│   └── resumes/
│       ├── Ava_Smith_Resume.pdf
│       ├── Ben_Carter_Resume.pdf
│       ├── Carlos_Lee_Resume.pdf
│       ├── Divya_Nair_Resume.pdf
│       ├── Emma_Wilson_Resume.pdf
│       └── Farah_Khan_Resume.pdf
│
├── output/
├── requirements.txt
├── .env.example
└── README.md
```



## Assignment Requirements Mapping



### Part A - Core File System Tools

`src/fs_tools.py` contains all four required tools.

#### 1. `read_file(filepath: str) -> dict`

- Reads PDF resume files.
- Uses `PyPDFLoader`.
- Extracts text from the PDF.
- Returns content and file metadata.
- Handles errors gracefully.



#### 2. `list_files(directory: str, extension: str = None) -> list`

- Lists files in a directory.
- Supports extension filtering.
- Returns:
  - name
  - path
  - extension
  - size
  - modified date



#### 3. `write_file(filepath: str, content: str) -> dict`

- Writes generated text to a file.
- Creates parent directories automatically.
- Returns success/failure status.



#### 4. `search_in_file(filepath: str, keyword: str) -> dict`

- Reads the PDF.
- Searches case-insensitively.
- Returns match count and surrounding context.



### Part B - LLM Integration

`src/llm_file_assistant.py`:

- Uses Cohere Command A.
- Uses `ClientV2`.
- Defines all four functions as structured tools.
- Allows the LLM to select tools from natural-language requests.
- Executes tool calls in Python.
- Sends tool results back to Cohere.
- Returns the final response.



## Architecture

```text
                    User Query
                         |
                         v
               +-------------------+
               | Cohere Command A  |
               +---------+---------+
                         |
                    Tool Calling
                         |
       +-----------------+------------------+
       |                 |                  |
       v                 v                  v
 read_file          list_files       search_in_file
       |                 |                  |
       +-----------------+------------------+
                         |
                         v
                  PyPDFLoader
                         |
                         v
                    PDF Text
                         |
                         v
                     Cohere
                         |
                         v
                   Final Answer

                  write_file
                       ^
                       |
             Generated Summary
```



## Setup

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```



### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Cohere API key.

### Windows PowerShell

```powershell
$env:COHERE_API_KEY="YOUR_COHERE_API_KEY"
```



### macOS/Linux

```bash
export COHERE_API_KEY="YOUR_COHERE_API_KEY"
```

Optional model configuration:

```bash
export COHERE_MODEL="command-a-03-2025"
```



## Run the Application

Run from the project root.

### Example 1 - List resumes

```bash
python src/llm_file_assistant.py "List all resumes in the documents/resumes folder"
```



### Example 2 - Read a resume

```bash
python src/llm_file_assistant.py "Read Ava Smith's resume"
```



### Example 3 - Search resumes

```bash
python src/llm_file_assistant.py "Find resumes mentioning Python experience"
```



### Example 4 - Create a summary

```bash
python src/llm_file_assistant.py "Create a summary file for documents/resumes/Ava_Smith_Resume.pdf"
```

The summary will be written to the `output` directory when the model follows
the configured instructions.

## Tool Calling Flow

For:

```text
Find resumes mentioning Python experience.
```

The expected flow is:

```text
User
  |
  v
Cohere
  |
  v
list_files()
  |
  v
PDF resume list
  |
  v
search_in_file()
  |
  v
PDF text + matching context
  |
  v
Cohere
  |
  v
Final answer
```

For:

```text
Create a summary file for Ava_Smith_Resume.pdf
```

the flow is:

```text
User
  |
  v
Cohere
  |
  v
read_file()
  |
  v
PyPDFLoader
  |
  v
Resume text
  |
  v
Cohere creates summary
  |
  v
write_file()
  |
  v
output/Ava_Smith_Resume_summary.txt
```

