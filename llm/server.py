"""
Standalone FastAPI server to host the LLM locally on the GPU.
Used when the main RAG application is running in AWS (or elsewhere) in "remote" mode.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import config
from llm.hf_model import HuggingFaceModel

app = FastAPI(title="DriveStream Local GPU Server")

# Initialize the model forcibly in local mode
print("Initializing LLM Server on Local GPU...")
llm = HuggingFaceModel(mode="local")
print("LLM Server is ready!")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = config.LLM_MAX_NEW_TOKENS


@app.post("/generate")
def generate_text(req: GenerateRequest):
    try:
        text = llm.generate(req.prompt, req.max_new_tokens)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
