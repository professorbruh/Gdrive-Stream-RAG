"""
Standalone FastAPI server to host the LLM locally on the GPU.
Used when the main RAG application is running in AWS (or elsewhere) in "remote" mode.
"""

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import uvicorn

import config
from llm.hf_model import HuggingFaceModel
from logger_setup import get_logger

logger = get_logger(__name__)

app = FastAPI(title="DriveStream Local GPU Server")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if config.LLM_API_KEY:
        if api_key != f"Bearer {config.LLM_API_KEY}":
            raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# Initialize the model forcibly in local mode
logger.info("Initializing LLM Server on Local GPU...")
llm = HuggingFaceModel(mode="local")
logger.info("LLM Server is ready!")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = config.LLM_MAX_NEW_TOKENS


@app.post("/generate")
def generate_text(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        text = llm.generate(req.prompt, req.max_new_tokens)
        return {"text": text}
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Admin is grinding"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
