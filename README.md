# Kasa OTA Monitor

This project provides an automated analysis and reporting tool for weekly OTA (Online Travel Agency) performance data, designed for Kasa's property portfolio. It processes multi-week Excel reports, computes key metrics, identifies alerts, highlights best performers, and generates actionable insights using a local LLM (Ollama). The results are returned as an enhanced Excel file and a JSON summary, suitable for API or Slack integration.

## Features
- **Automated Metrics Calculation:** Computes conversion rates, search visibility, stability index, and week-over-week (WoW) changes for each listing.
- **Alert Classification:** Flags listings with significant performance drops (P1/P2/P3) and provides root cause suggestions.
- **Best Performer Identification:** Highlights listings with strong, stable performance and extracts replicable strategies.
- **LLM Insights:** Uses a local Ollama LLM to generate a narrative summary and recommendations based on the data.
- **API Endpoint:** FastAPI endpoint for uploading Excel files and receiving enhanced reports and insights.

## How It Works
1. **Upload:** Submit a multi-sheet Excel file (each sheet = one week) via the `/analyze` API endpoint.
2. **Processing:**
   - Loads and merges all weeks' data.
   - Computes metrics and stability for each listing.
   - Calculates week-over-week changes.
   - Classifies alerts and finds best performers for the latest week.
   - Calls Ollama (local LLM) for a narrative summary.
3. **Output:**
   - Returns a new Excel file with added sheets: per-week metrics, alerts, best performers, and LLM insights.
   - Returns a JSON summary with grouped alerts and the LLM-generated narrative.

## Requirements
- Python 3.8+
- [Ollama](https://ollama.com/) running locally (default: http://localhost:11434, with the `gemma3:latest` model pulled)
- See `requirements.txt` for Python dependencies

## Installation
```bash
# Clone the repo and enter the directory
cd kasa-ota-monitor

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the API
Start the FastAPI server (default: http://127.0.0.1:8000):

```bash
uvicorn api:app --reload
```

## API Usage
Send a POST request to `/analyze` with a multi-week Excel file:

- **Endpoint:** `POST /analyze`
- **Form field:** `file` (the Excel file to analyze)
- **Response:**
  - `file_base64`: base64-encoded enhanced Excel file
  - `file_name`: suggested output filename
  - `last_week`, `previous_week`: week labels
  - `alerts`: grouped alert listings
  - `llm_insights`: narrative summary (for email or Slack)

Example using `curl`:
```bash
curl -F "file=@your_input.xlsx" http://127.0.0.1:8000/analyze
```

## Notes
- Ollama must be running locally with the required model before using the API.
- The tool is designed for internal analytics and reporting workflows.

## License
MIT License
