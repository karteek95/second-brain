import asyncio
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from second_brain.infrastructure.llms import build_chat_model
from second_brain.infrastructure.observability import build_trace_sink

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


llm_config = {
    "provider": "ollama",
    "model": "qwen2.5:7b-instruct",
    "temperature": 0.1
}
chat_model = build_chat_model(llm_config)


async def real_llm_stream(question: str, trace_sink):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": question}
    ]

    full_response = ""

    # 1. Open the observation wrapper (starts the latency timer natively)
    with trace_sink.observe_stream(
            event="chat_completion",
            input_data=question,
            metadata={"model": llm_config["model"]}
    ) as obs:
        # 2. Yield tokens to the UI (this takes real time)
        for token in chat_model.generate_stream(messages):
            full_response += token
            yield token
            await asyncio.sleep(0.01)

            # 3. Save the final compiled response into the trace right before closing
        obs.update(output=full_response)

    # Natively, once the block exits, the trace captures the end time and calculates latency!


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    trace_config = {
        "run_name": "web_ui_chat"
    }
    trace_sink = build_trace_sink(trace_config)

    return StreamingResponse(
        real_llm_stream(req.question, trace_sink),
        media_type="text/event-stream"
    )