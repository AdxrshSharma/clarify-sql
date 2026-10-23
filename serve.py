"""Step 6: serve the clarification-aware text-to-SQL model with FastAPI.

  BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct ADAPTER_DIR=outputs/lora uvicorn serve:app --port 8000

Flow:
  POST /ask {"question": "How did DV360 perform in July 2026?"}
    -> {"status": "needs_clarification", "question": "Which metric ...?"}
  POST /ask {"question": "...same...", "clarification": "ROAS"}
    -> {"status": "ok", "sql": "...", "columns": [...], "rows": [...]}
"""
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backends import HFBackend
from common import parse_output
from sql_guard import execute_safe

app = FastAPI(title="Clarification-aware Text-to-SQL")
backend = HFBackend(os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
                    os.environ.get("ADAPTER_DIR"))


class Ask(BaseModel):
    question: str
    clarification: Optional[str] = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(req: Ask):
    text = req.question if not req.clarification else f"{req.question}\nClarification: {req.clarification}"
    raw, _, _ = backend.predict(text)
    kind, payload = parse_output(raw)
    if kind == "clarify":
        return {"status": "needs_clarification", "question": payload}
    try:
        columns, rows = execute_safe(payload)
    except Exception as e:  # invalid or unsafe SQL never reaches the user as data
        raise HTTPException(status_code=422, detail={"error": str(e), "sql": payload})
    return {"status": "ok", "sql": payload, "columns": columns, "rows": rows}
