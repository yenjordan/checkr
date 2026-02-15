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
    page_texts = []
    page_layouts = []

    for page in document.pages:
        segments = []
        lines_data = []

        # Use lines for layout data, fall back to paragraphs
        layout_items = page.lines if page.lines else page.paragraphs

        for item in layout_items:
            for segment in item.layout.text_anchor.text_segments:
                start = segment.start_index or 0
                end = segment.end_index
                if end > start:
                    text = document.text[start:end]
                    segments.append((start, end))

                    verts = item.layout.bounding_poly.normalized_vertices
                    if len(verts) >= 4:
                        lines_data.append({
                            "t": text.rstrip("\n"),
                            "x": round(verts[0].x, 4),
                            "y": round(verts[0].y, 4),
                            "w": round(max(verts[1].x - verts[0].x, 0.01), 4),
                            "h": round(max(verts[2].y - verts[0].y, 0.001), 4),
                        })

        if segments:
            page_start = min(s[0] for s in segments)
            page_end = max(s[1] for s in segments)
            page_texts.append(document.text[page_start:page_end])
        else:
            page_texts.append("")

        page_layouts.append(lines_data)

    return {
        "full_text": document.text,
        "pages": len(document.pages),
        "page_texts": page_texts,
        "page_layouts": page_layouts,
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