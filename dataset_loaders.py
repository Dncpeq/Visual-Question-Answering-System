


# libs
import json, datasets, os
from pdf2image import convert_from_bytes
from torch.utils.data import Dataset
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from dataclasses import dataclass
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from typing import Callable, Dict, Any
from enum import Enum

# project libs
from utils import get_paths

# adaptor output format requirements
# [ # batch
#     [    # sequence: one sequence represent one session/message.
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": ...
#                 },
#                 {
#                     "type": "text",
#                     "text": ...
#                 },
#                 ...  
#             ] 
#         },
#         {
#             "role": "assistant",
#             "content": [
#                 ...
#             ]
#         },
#         ...
#     ],
#     ...
# ] 


class Role(Enum):
    user = "user"
    assistant = "assistant"

def form_message_item(role, content):
    return {
        "role": role,
        "content": content
    }


def form_content(items):
    '''
    input format
    [
        {"text": ...},
        {"image": ...},
        ...
    ]
    '''
    def content_template(content, type):
        if type == "text":
            content = {
                "type": "text", 
                "text": content
            }
        elif type == "image":
            content = {
                "type": "image",
                "image": content
            }
        elif type == "video":
            content = {
                "type": "video",
                "video": content
            }
        return content
    
    content = []
    for item in items:
        if "text" in item:
            content.append(content_template(item["text"], "text"))
        if "image" in item:
            content.append(content_template(item["image"], "image"))
        if "video" in item:
            content.append(content_template(item["video"], "video"))
    return content


def form_item(role, contents):
    return form_message_item(
        role,
        form_content(contents)
    )

# jackyhate_text2image dataset is raw files, need to construct
class jackyhate_text2image(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.files = []
        for file in get_paths(self.path, file_only=True, suffixes=["jpg"]):
            file = os.path.splitext(file)
            # if file not in self.files:
            self.files.append(file[-2])
        self._len = len(self.files)

    def __len__(self):
        return self._len
    
    def __getitem__(self, index):
        file = self.files[index]
        with open(file+".json", 'r', encoding="utf-8") as f:
            content = f.read()
            content = json.loads(content)["prompt"]
        img = f"{file}.jpg"
        # img = Image.open(img)
        return {
            "image": img,
            "text": content
        }
    
    def __iter__(self):
        for i in range(self._len):
            yield self.__getitem__(i)

def jackyhate_text2image_data_loader(dataset_root):
    dataset = jackyhate_text2image(dataset_root)
    dataset_features = datasets.Features({"image": datasets.Image(), "text": datasets.Value("string")})
    dataset = datasets.Dataset.from_list(
        [data for data in dataset], 
        features=dataset_features).with_format("torch", columns=[], output_all_columns=True)
    return dataset

def adaptor_jackyhate_text2image(batch, processor: Qwen2_5_VLProcessor):
    sessions = []
    for item in batch:
        session = []
        session.append(
            form_item(
                Role.user.value,
                [{"image": item["image"]} ] #, {"text": get_caption_question()}])
            )
        ) # object is PIL.Image.Image

        session.append(
            form_item(
                Role.user.value,
                [{"text": item["text"]}]
            )
        ) # this is image description text
        # session.append({"example_ids": item["example_ids"]}) # assume Trainer will add this attribute
        sessions.append(session)
    
    return sessions



def mOSCAR_data_loader(dataset_root, data_files = None, streaming = False):
    '''
    examples:
    dataset
    DatasetDict({
        train: Dataset({
            features: ['images', 'text', 'metadata'],
            num_rows: 98982
        })
    })

    dataset["train"]
    output:
    DatasetDict({
        train: Dataset({
            features: ['images', 'text', 'metadata'],
            num_rows: 98982
        })
    })
    '''
    data_files = data_files if data_files else f"{dataset_root}/*.parquet"
    dataset = load_dataset("parquet", data_files=data_files, streaming=streaming)
    return dataset

def adaptor_mOSCAR_text(batch, processor: Qwen2_5_VLProcessor):
    """
    mOSCAR is a plain text dataset, there is no QA format
    """
    sessions  = []
    for item in batch:
        session = []
        # texts: List[{"text": "...", 'text_idx': '#000000'}, ...]
        texts = item["text"]
        text = "\n".join([text["text"] for text in texts])

        # the format is to utilize the processor apply template for format and tokenization
        # role is no use 
        session.append(
            form_item(
               Role.user.value, 
               [{"text": text}]
            )
        )
        sessions.append(session)
    return sessions


def Docmatix_data_loader(dataset_root, data_files = None, streaming = False):
    data_files = data_files if data_files else f"{dataset_root}/*.parquet"
    dataset = load_dataset("parquet", data_files=data_files, streaming=streaming)
    return dataset


def adaptor_Docmatix_data(batch, processor):
    sessions = []
    for item in batch:
        images, texts = item["images"], item["texts"]
        session = []
        for text in texts:
            session.append(
                form_item(
                    Role.user.value,
                    [{"text": text["user"]}]
                )
            )
            session.append(
                form_item(
                    Role.assistant.value,
                    [{"text": text["assistant"]}]
                )
            )
        session[0]["content"] = form_content([{"image": img} for img in images]) + session[0]["content"]
        sessions.append(session)
    return sessions



def adaptor_Docmatix_pdf(batch, processor):
    sessions = []
    for item in batch:
        pdf_bytes, qas = item["pdf"], item["texts"]
        session = []
        pdf_images = convert_from_bytes(pdf_bytes)

        session.append(
            form_item(
                Role.user.value, 
                [{"image": img} for img in pdf_images] + [{"text": qas[0]["user"]}]
            ) 
            # we may add random to switch the sequence of image and question
        )
        session.append(
            form_item(
                Role.assistant.value, 
                [{"text": qas[0]["assistant"]}]
            )
        )

        for qa in qas[1:]:
            session.append(
                form_item(
                    Role.user.value,
                    [{"text": qa["user"]}]
                )
            )
            session.append(
                form_item(
                    Role.assistant.value,
                    [{"text": qa["assistant"]}]
                )
            )
        sessions.append(session)
    return sessions



def Chinese_DeepSeek_R1_Distill_dataset_loader(dataset_root, data_files = None, streaming = False):
    data_files = data_files if data_files else f"{dataset_root}/*.jsonl"
    dataset = load_dataset("json", data_files=data_files, streaming=streaming)
    return dataset

def adaptor_DeepSeek_R1_Distill(batch, processor):
    sessions = []
    for input, reasoning_content, content in zip(batch["input"], batch["reasoning_content"], batch["content"]):
        session = []
        session.append(
            form_item(
                Role.user.value,
                [{"text": input}]
            )
        )
        session.append(
            form_item(
                Role.assistant.value,
                [{"text": reasoning_content + "\n" + content}]
            )
        )
        sessions.append(session)
    return sessions


@dataclass
class MultiwayDataCollator():
    """
    Data collator for Multiway model that handles processing of text, images, and videos.
    """
    processor: Qwen2_5_VLProcessor
    max_length: int = None
    chat_template: str = None
    call_back: Callable[[Dict], None] = None

    def __call__(self, features):
        features = self.adaptor(features, self.processor)
        return self.collate_fn(features, self.processor, self.max_length, self.chat_template, self.call_back)

    @staticmethod
    def adaptor(batch, processor):
        return batch

    @staticmethod
    def collate_fn(batch, processor: Qwen2_5_VLProcessor, max_length = None, chat_template = None, call_back = None):
        """
        Collate function for DataLoader that processes batch inputs through the processor.
        """
        text_inputs = processor.apply_chat_template(batch, chat_template=chat_template, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(batch)
        batched_inputs = processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation= max_length is not None,
            max_length = max_length,
            return_tensors="pt"
        )
        
        if call_back:
            call_back(batched_inputs)

        return batched_inputs





