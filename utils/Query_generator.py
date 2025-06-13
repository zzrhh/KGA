import os
import json
import torch
import jsonlines
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import AsyncOpenAI
import asyncio


class QAPathGenerator:
    def __init__(self, model_id: str, batch_size: int = 5):
        self.model_id = model_id
        self.batch_size = batch_size
        self.is_llama = "llama" in model_id.lower() or Path(model_id).exists()

        if self.is_llama:
            print("🔧 Loading LLaMA model...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb_config
            )
            self.model.eval()
            self.device = self.model.device
        else:
            print("🌐 Using OpenAI GPT model...")
            self.client = AsyncOpenAI(api_key= os.getenv("OPENAI_API_KEY"),
                base_url= os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
            self.tokenizer = None  # not needed


    def _build_prompt(self, path):
        head_entity = path[0][0]
        return f"""You are a multi-hop question generation assistant for knowledge graph paths.

    Your task is to generate a natural language question based on a sequence of triples (path), such that:
    - Only the **head entity** "{head_entity}" is mentioned in the question.
    - Do not mention any intermediate or tail entities by name.
    - The question must follow each step of the path explicitly, and each predicate should be reflected clearly using its exact wording from the path whenever possible. Avoid paraphrasing or replacing predicates with synonyms unless grammatically necessary.
    - The question must lead to a **single answer**, which is the tail of the last triple.
    - Avoid vague, abstract, or overly verbose phrasing.


    ### Example:

    Path:
    [["Park Row", "was directed by", "Samuel Fuller"], ["Samuel Fuller", "directed", "I Shot Jesse James"], ["I Shot Jesse James", "was released in year", "1949"]]

    Output:
    {{
    "question": "What year was the film directed by the director of Park Row released?"
    }}

    ---

    Now generate a question for the following path:

    Path:
    {json.dumps(path, ensure_ascii=False)}

    Output:
    """

    async def generate(self, path):
        prompt = self._build_prompt(path)

        if self.is_llama:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that follows few-shot examples to generate multi-hop QA from paths."},
                {"role": "user", "content": prompt}
            ]
            chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=500,
                    do_sample=False,
                    temperature=0.7,
                    eos_token_id=[
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
                )
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

            del inputs, input_ids, outputs
            torch.cuda.empty_cache()

        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that follows few-shot examples to generate multi-hop QA from paths."},
                {"role": "user", "content": prompt}
            ]
            try:
                response_obj = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1024 
                )
                response = response_obj.choices[0].message.content.strip()
            except Exception as e:
                return {
                    "question": "[ERROR]",
                }

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "raw_output": response
            }

    async def generate_from_jsonl(self, input_path: str, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        results = []
        sem = asyncio.Semaphore(self.batch_size)

        async def process_item(obj):
            path = obj.get("path")
            if not path:
                return None
            async with sem:
                return await self.generate(path)

        # 读入并创建任务
        with jsonlines.open(input_path, "r") as reader:
            items = [obj for obj in reader]

        tasks = [process_item(obj) for obj in items]

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating QA"):
            result = await future
            if result:
                results.append(result)

        # 写入
        with jsonlines.open(output_path, "w") as writer:
            for item in results:
                writer.write(item)

        print(f"✅ Saved to {output_path}")


