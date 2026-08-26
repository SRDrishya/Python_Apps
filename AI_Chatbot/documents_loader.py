from bs4 import BeautifulSoup
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook

def load_documents(self):
    """Load supported documents into a common representation."""

    if not self.input_file.is_dir():
        with open(
            self.input_file,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    documents = []

    for file_path in sorted(self.input_file.iterdir()):

        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                documents.extend(
                    self.load_pdf(file_path)
                )

            elif suffix == ".txt":
                documents.extend(
                    self.load_txt(file_path)
                )

            elif suffix == ".docx":
                documents.extend(
                    self.load_docx(file_path)
                )

            elif suffix == ".md":
                documents.extend(
                    self.load_txt(file_path)
                )

            elif suffix == ".csv":
                documents.extend(
                    self.load_csv(file_path)
                )

            elif suffix == ".json":
                documents.extend(
                    self.load_json(file_path)
                )

            elif suffix in [".html", ".htm"]:
                documents.extend(
                    self.load_html(file_path)
                )

            elif suffix == ".xlsx":
                documents.extend(
                    self.load_xlsx(file_path)
                )

            elif suffix == ".pptx":
                documents.extend(
                    self.load_pptx(file_path)
                )

            else:
                print(
                    f"Skipping unsupported file: "
                    f"{file_path.name}"
                )

        except Exception as e:
            print(
                f"Error loading {file_path.name}: {e}"
            )

    return documents

def load_pdf(self, file_path):

    documents = []

    reader = PdfReader(str(file_path))

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append({
            "text": text,
            "metadata": {
                "source": file_path.name,
                "file_type": "pdf",
                "page": page_number
            }
        })

    return documents

def load_txt(self, file_path):

    text = file_path.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        return []

    return [{
        "text": text,
        "metadata": {
            "source": file_path.name,
            "file_type": file_path.suffix.lower().replace(".", ""),
            "page": None
        }
    }]

def load_docx(self, file_path):

    doc = Document(str(file_path))

    parts = []

    for paragraph in doc.paragraphs:

        text = paragraph.text.strip()

        if text:
            parts.append(text)

    # Include tables
    for table in doc.tables:

        for row in table.rows:

            cells = [
                cell.text.strip()
                for cell in row.cells
            ]

            row_text = " | ".join(
                cell for cell in cells
                if cell
            )

            if row_text:
                parts.append(row_text)

    text = "\n".join(parts)

    if not text.strip():
        return []

    return [{
        "text": text,
        "metadata": {
            "source": file_path.name,
            "file_type": "docx",
            "page": None
        }
    }]

def load_csv(self, file_path):

    import csv

    documents = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row_number, row in enumerate(
            reader,
            start=1
        ):

            parts = []

            for key, value in row.items():

                if value is None:
                    continue

                value = str(value).strip()

                if value:
                    parts.append(
                        f"{key}: {value}"
                    )

            text = "\n".join(parts)

            if not text.strip():
                continue

            documents.append({
                "text": text,
                "metadata": {
                    "source": file_path.name,
                    "file_type": "csv",
                    "page": None,
                    "row": row_number
                }
            })

    return documents

def load_json(self, file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    text = json.dumps(
        data,
        indent=2,
        ensure_ascii=False
    )

    if not text.strip():
        return []

    return [{
        "text": text,
        "metadata": {
            "source": file_path.name,
            "file_type": "json",
            "page": None
        }
    }]

def load_html(self, file_path):

    html = file_path.read_text(
        encoding="utf-8"
    )

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # Remove things that shouldn't enter RAG
    for element in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header"
    ]):
        element.decompose()

    text = soup.get_text(
        separator="\n"
    )

    text = self.normalize_text(text)

    if not text.strip():
        return []

    return [{
        "text": text,
        "metadata": {
            "source": file_path.name,
            "file_type": "html",
            "page": None
        }
    }]

