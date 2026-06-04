"""
ARGUS FastAPI Backend
REST + WebSocket endpoints for chart analysis.
"""
import io
import base64
from typing import Optional
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from argus_core import ArgusOrchestrator

app = FastAPI(
    title="ARGUS — AI Visual Market Analyst",
    description="Upload chart screenshots. Get trading intelligence.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One global orchestrator per server instance
argus = ArgusOrchestrator()


# ─── Models ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class AnalyzeRequest(BaseModel):
    image_base64: str
    question: Optional[str] = "Analyze this chart and give me your assessment."


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models": argus.list_available_models()}


@app.post("/analyze")
async def analyze_chart(file: UploadFile = File(...), question: str = "Analyze this chart."):
    """Upload a chart image file and receive full analysis."""
    content = await file.read()
    img_array = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    argus.reset()
    result = argus.analyze(img, question, stream=False)

    return {
        "signal": result.signal.action,
        "conviction": result.signal.conviction,
        "risk_rating": result.signal.risk_rating,
        "risk_reward": result.signal.risk_reward_ratio,
        "entry_zone": result.signal.entry_zone,
        "stop_loss": result.signal.stop_loss_zone,
        "target": result.signal.target_zone,
        "reasoning": result.signal.reasoning_points,
        "warnings": result.signal.warnings,
        "market_summary": result.summary.to_dict(),
        "llm_response": result.llm_response,
        "processing_ms": result.processing_time_ms,
    }


@app.post("/analyze/base64")
async def analyze_base64(req: AnalyzeRequest):
    """Analyze from a base64-encoded image string."""
    try:
        img_bytes = base64.b64decode(req.image_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    argus.reset()
    result = argus.analyze(img, req.question, stream=False)
    return {
        "llm_response": result.llm_response,
        "signal": result.signal.action,
        "market_summary": result.summary.to_dict(),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """Follow-up question about the currently analyzed chart."""
    response = argus.chat(req.message, stream=False)
    return {"response": response}


@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    """
    WebSocket endpoint for streaming analysis.
    Send: JSON {"image_base64": "...", "question": "..."}
    Receive: streaming tokens
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        img_bytes = base64.b64decode(data["image_base64"])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        question = data.get("question", "Analyze this chart.")

        argus.reset()
        for token in argus.analyze(img, question, stream=True):
            await websocket.send_text(token)
        await websocket.send_text("[DONE]")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"[ERROR] {e}")
