# Multimodal RAG Test Dataset

This dataset is designed to test a RAG pipeline against multiple
document types and content types.

## Files

- employee_handbook.docx
  - Text
  - Table
  - Embedded image

- payment_policy.pdf
  - Text
  - Table
  - Architecture diagram

- product_catalog.csv
  - Structured tabular data

- customers.json
  - Structured JSON data

- company_info.html
  - HTML text
  - Image reference

- financial_report.xlsx
  - Spreadsheet table
  - Bar chart

- architecture.png
  - Architecture diagram

- rag_test_questions.json
  - Direct questions
  - Table questions
  - Image questions
  - Chart questions
  - Cross-document questions
  - Unanswerable questions

## Suggested metadata

Your ingestion pipeline should preserve metadata such as:

source
file_type
page
sheet
section
element_type
table_id
image_id
chunk_id

## Important

After copying these documents into your RAG document directory,
run your ingestion process before starting the chatbot.

Example:

python ingest.py

Then:

python main.py
