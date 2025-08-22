
import asyncio, time, json, torch
from threading import Thread
from transformers.generation import AsyncTextIteratorStreamer
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from modeling import MultiwayQwen2_5VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class ModelService:
    def __init__(
        self,
        model: MultiwayQwen2_5VLForConditionalGeneration,
        processor: Qwen2_5_VLProcessor,
        queue: asyncio.Queue,
        max_seq_len: int = None,
    ):
        self.max_seq_len = max_seq_len
        self.model = model
        self.processor = processor
        self.queue = queue

    async def start_service(self):
        while True:
            request = await self.queue.get()    # wait for request

            task_data, response_queue = request

            await self._inference(task_data, response_queue, task_data["max_new_tokens"])


    async def _inference(self, task_data, response_queue: asyncio.Queue, max_new_tokens = 128):
        task_id = task_data["id"]
        message = task_data["messages"]

        input_text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        input_images, input_videos = process_vision_info(message)

        inputs = self.processor(
            text=input_text,
            images=input_images,
            videos=input_videos,
            padding=True,
            return_tensors="pt",
            truncation=self.max_seq_len is not None,
            max_length=self.max_seq_len
        )

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        streamer = AsyncTextIteratorStreamer(
            self.processor.tokenizer, 
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        worker = Thread(
            target = self.model.generate,
            kwargs = inputs | {"streamer": streamer, "max_new_tokens": max_new_tokens}
        )
        worker.start()

        # try:
        #     async for text in streamer:
        #         response_queue.put_nowait(text)
        # finally:
        #     response_queue.put_nowait(None)

        try:
            async for text in streamer:
                if text:
                    # print(text, end='')
                    chunk = {
                        "id": task_id, # You'll need to pass task_id to _inference
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "qwen2.5-vl", # Or get from request
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]
                    }
                    response_queue.put_nowait(json.dumps(chunk)) # Put JSON string

        # ... inside _inference finally block ...
        finally:
            # print()
            finish_chunk = {
                "id": task_id, # Pass task_id
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "qwen2.5-vl", # Or get from request
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            response_queue.put_nowait(json.dumps(finish_chunk))
            response_queue.put_nowait(None) # Signal end AFTER the final chunk









