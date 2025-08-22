
import torch, uvicorn, os, asyncio, json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator, Optional, Literal, Union

from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from model_service import ModelService


STATIC_DIR = "dist"
MODEL_DIR = "Qwen2.5-VL-3B-Instruct"
MAX_SEQ_LEN = 1024
device_map = 'cuda'

request_queue = asyncio.Queue()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextContentItem(BaseModel):
    type: Literal["text"]
    text: str

class ImageContentItem(BaseModel):
    type: Literal["image"]
    image: str

class VideoContentItem(BaseModel):
    type: Literal["video"]
    video: str

class Message(BaseModel):
    role: str
    content: List[Union[VideoContentItem, ImageContentItem, TextContentItem]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    max_new_tokens: Optional[int] = 512 # Add generation params


async def stream_generator(response_queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    # while True:
    #     content = await response_queue.get()
    #     if content is None: break
    #     yield content
    # response_queue.task_done()

    while True:
        content = await response_queue.get()
        if content is None:
            yield "data: [DONE]\n\n" # OpenAI specific end signal (optional but common)
            break
        yield f"data: {content}\n\n" # Format as SSE
    response_queue.task_done()


@app.on_event("startup")
async def startup_event():
    global MODEL_DIR, MAX_SEQ_LEN, device_map
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map=device_map)
    processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_DIR)
    model_service = ModelService(model, processor, request_queue, MAX_SEQ_LEN)
    asyncio.create_task(model_service.start_service())


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.stream:
        raise HTTPException(status_code=400, detail="Only streaming requests are supported.")

    response_queue = asyncio.Queue()
    task_id = f"chatcmpl-{asyncio.get_event_loop().time()}"
    task_data = request.model_dump()
    task_data['id'] = task_id

    await request_queue.put((task_data, response_queue))

    return StreamingResponse(
        stream_generator(response_queue),
        media_type="text/event-stream",
        headers={'Cache-Control': 'no-cache'}
    )

app.mount(
    "/", 
    StaticFiles(directory=STATIC_DIR, html=True), 
    name="static",
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)