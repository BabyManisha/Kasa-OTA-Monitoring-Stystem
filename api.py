# api.py
import base64

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from analysis import build_insights_excel

app = FastAPI(title="Kasa OTA Alert Service")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    n8n will call this endpoint with the weekly Excel file.
    We return:
      - file_base64: the new *_insights.xlsx (as base64-encoded binary)
      - file_name: suggested filename
      - last_week / previous_week
      - alerts: { "1": [...], "2": [...], "3": [...] }
      - llm_insights: long text formatted for email (for sending as an email body) don't say anything like okay, here is the analysis, let me know if you need anything else, etc.
    """
    content = await file.read()

    excel_out, summary = build_insights_excel(content)

    b64_excel = base64.b64encode(excel_out).decode("utf-8")
    base_name = file.filename.rsplit(".", 1)[0]
    out_name = f"{base_name}_insights.xlsx"

    return JSONResponse({
        "file_base64": b64_excel,
        "file_name": out_name,
        **summary,
    })
