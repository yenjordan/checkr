import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account
from supabase import create_client
import sys

load_dotenv()

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT_ID"]
LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]
PROCESSOR_ID = os.environ["GOOGLE_DOCUMENT_AI_PROCESSOR_ID"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
SERVICE_ACCOUNT_JSON = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]


def extract_pdf(pdf_path: str) -> documentai.Document:
    opts = ClientOptions(
        api_endpoint=f"{LOCATION}-documentai.googleapis.com",
    )
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(SERVICE_ACCOUNT_JSON)
    )
    client = documentai.DocumentProcessorServiceClient(
        client_options=opts, credentials=credentials
    )
    resource_name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)
    pdf_bytes = Path(pdf_path).read_bytes()
    raw_document = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
    request = documentai.ProcessRequest(
        name=resource_name,
        raw_document=raw_document,
        imageless_mode=True,
    )
    result = client.process_document(request=request)
    return result.document

def parse_document(document: documentai.Document) -> dict:
    return {
        "full_text": document.text,
        "pages": len(document.pages),
        "entities": [
            {"type": e.type_, "text": e.mention_text, "confidence": e.confidence}
            for e in document.entities
        ],
    }

def upload_to_supabase(data: dict, table: str = "papers") -> dict:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    result = supabase.table(table).insert(data).execute()
    return result.data

def process_paper(pdf_path: str) -> dict:
    document = extract_pdf(pdf_path)
    data = parse_document(document)
    data["source_filename"] = Path(pdf_path).name
    uploaded = upload_to_supabase(data)
    return uploaded

if __name__ == "__main__":
    result = process_paper(sys.argv[1])