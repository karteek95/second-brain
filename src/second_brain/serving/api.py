import asyncio
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from second_brain.infrastructure.config import load_config
from second_brain.infrastructure.vector_stores import build_vector_index
from second_brain.infrastructure.llms import build_chat_model
from second_brain.infrastructure.observability import build_trace_sink
from second_brain.application import AgenticRagUseCase

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


# REQUIREMENT 4: Dynamic Config loading
config_path_str = os.getenv("SECOND_BRAIN_CONFIG", "configs/chroma_hybrid_bm25.yaml")
config = load_config(Path(config_path_str))

vector_index = build_vector_index(config["vector_index"])
chat_model = build_chat_model(config["llm"])


async def real_llm_stream(question: str, trace_sink):
    agent = AgenticRagUseCase(
        vector_index=vector_index,
        chat_model=chat_model,
        trace_sink=trace_sink
    )

    with trace_sink.observe_stream(
            event="chat_completion",
            input_data=question,
            metadata={"model": config["llm"]["model"]}
    ) as obs:

        # Offload the heavy, synchronous blocking task to a background thread
        # This prevents the FastAPI server from freezing and timing out the UI
        full_response = await asyncio.to_thread(agent.answer, question)

        # Debugging: Ensure the backend actually generated text
        print(f"--- BACKEND GENERATED ---\n{full_response}\n-------------------------")

        # Fallback if the string is completely empty
        if not full_response.strip():
            full_response = "Error: The model returned an empty string. Check the backend terminal."

        words = full_response.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.01)

        obs.update(output=full_response)


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # Use observability config from yaml
    trace_sink = build_trace_sink(config["observability"])

    return StreamingResponse(
        real_llm_stream(req.question, trace_sink),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("second_brain.serving.api:app", host="127.0.0.1", port=8000, reload=True)