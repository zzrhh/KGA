import torch
import math
import torch.nn.functional as F
from typing import List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from openai import AsyncOpenAI
from utils.Cypher_Generation_Prompt import generate_cypher_prompt_OGBN,generate_cypher_prompt_meta_QA,generate_cypher_prompt_2wiki_webquestions,generate_QA_prompt,generate_QA_prompt_Praphra,generate_QA_prompt_Prompt_modi
import os
import asyncio
class TokenProbabilityExtractor:
    def __init__(self, model_name_or_path: str):
        """
        初始化 TokenProbabilityExtractor
        
        Args:
            model_name_or_path: 模型名称或路径，可以是:
                - OpenAI模型名称 (如 "gpt-4o-mini")
                - Hugging Face模型路径 (如 "./Model/llama-3-8b-Instruct")
        """
        self.model_name = model_name_or_path
        self.is_openai = "gpt" in model_name_or_path.lower()
        self.is_llama = "llama" in model_name_or_path.lower()
        
        if self.is_openai:
            self.client = AsyncOpenAI(  # 替换为异步客户端
                api_key= "sk-e0uZpTERY4UsnaEK7b6e7f96664548D8A618D06dB9A883Ad",
                base_url= "https://api-2.xi-ai.cn/v1"   
            )
        if self.is_llama:
            model_id = "meta-llama/Llama-3.1-8b-instruct"
            # ✅ 设置 4-bit 量化配置
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant= False,
                bnb_4bit_compute_dtype=torch.bfloat16  # 可选 float16 视硬件支持
            )

            # ✅ 加载分词器和量化模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb_config
            )
            self.device = self.model.device
        
    def get_cypher_queries(self, query: str,target_dataset = "OGBN-MAG") -> List[str]:
        if target_dataset == "OGBN-MAGMAG":
            prompt = generate_cypher_prompt_OGBN(query)
        elif target_dataset == "meta_qa":
            prompt = generate_cypher_prompt_meta_QA(query)
        elif target_dataset == "2wiki":
            prompt = generate_cypher_prompt_2wiki_webquestions(query)
        if self.is_openai:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "system", "content":"You are a helpful assistant in transforming users questions into cypher queries, based on provided shcema of the graph.",
                    "role": "user", "content": prompt}],
                stream=False,
                temperature=0.0,
            )
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

                # 构造 chat prompt 文本
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

                # 编码
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            prompt_len = input_ids.shape[1]

            with torch.no_grad():
                gen_outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=250,
                    do_sample=True,
                    temperature=0.0,
                    top_p=0.9,
                    eos_token_id=[
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
                )

            response = self.tokenizer.decode(
                gen_outputs[0][prompt_len:], skip_special_tokens=True
            )
        return response
            
    def get_QA_answer(self, question: str, cypher_answer: str):
        prompt = generate_QA_prompt(question, cypher_answer)

        if self.is_openai:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers the user's question using a provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                stream=False,
                logprobs=True,
                top_logprobs=5
            )

            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            logprobs = [entry.logprob for entry in top_logprobs[:5]]
            while len(logprobs) < 5:
                logprobs.append(0.0)

            feature = logprobs
            linear_probs = [math.exp(lp) for lp in feature]

            return response.choices[0].message.content, top_logprobs, linear_probs, feature

        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

            # 构造 chat prompt 文本
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 编码
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            prompt_len = input_ids.shape[1]

            # Step 1️⃣ 获取首 token 的 logits → top-5 logprobs
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                first_token_logits = logits[0, -1, :]
                log_probs = torch.log_softmax(first_token_logits, dim=-1)

                topk = torch.topk(log_probs, k=5)
                topk_ids = topk.indices.tolist()
                topk_logps = topk.values.tolist()
                topk_tokens = [self.tokenizer.decode([tid]) for tid in topk_ids]

            # Step 2️⃣ 使用 generate 获取完整回答
            with torch.no_grad():
                gen_outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=[
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
                )

            # Step 3️⃣ 构造 feature 向量
            feature = topk_logps[:5]
            while len(feature) < 5:
                feature.append(0.0)
            linear_probs = [math.exp(lp) for lp in feature]

            # 解码生成内容
            full_output_text = self.tokenizer.decode(
                gen_outputs[0][prompt_len:], skip_special_tokens=True
            )

            # 输出调试信息
            print("\n🤖 模型回答：")
            print(full_output_text)

            print("\n🔍 Top-5 候选（回答第一个 token 的预测）:")
            for tok, logp in zip(topk_tokens, topk_logps):
                print(f"Token: {repr(tok):<12} | log_prob: {logp:.4f}")

            return full_output_text, list(zip(topk_tokens, topk_logps)), linear_probs, feature


        
    async def get_cypher_queries_async(self, query: str, target_dataset="OGBN-MAG") -> str:
        if target_dataset == "OGBN-MAG":
            prompt = generate_cypher_prompt_OGBN(query)
        elif target_dataset == "meta_qa":
            prompt = generate_cypher_prompt_meta_QA(query)
        elif target_dataset == "2wiki":
            prompt = generate_cypher_prompt_2wiki_webquestions(query)

        if self.is_openai:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant in transforming users questions into cypher queries for strictly linear reasoning path."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            return response

        elif self.is_llama:
            loop = asyncio.get_event_loop()

            def llama_generate():
                messages = [
                    {"role": "system", "content": "You are a helpful assistant in transforming users questions into cypher queries for strictly linear reasoning path."},
                    {"role": "user", "content": prompt},
                ]
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=500,
                        do_sample=False,
                        temperature=0.0,
                        eos_token_id=[
                            self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ],
                    )

                response_text = self.tokenizer.decode(
                    outputs[0][input_ids.shape[1]:], skip_special_tokens=True
                )

                # ✅ 清理显存
                del inputs, input_ids, outputs
                torch.cuda.empty_cache()

                return response_text.strip()

            return await loop.run_in_executor(None, llama_generate)

        else:
            return self.get_cypher_queries(query)  # fallback 同步版本


    async def get_QA_answer_async(self, question: str, cypher_answer: str):
        prompt = generate_QA_prompt(question, cypher_answer)

        if self.is_openai:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers the user's question using a provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                logprobs=True,
                top_logprobs=5
            )

            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            logprobs = [entry.logprob for entry in top_logprobs[:5]]
            while len(logprobs) < 5:
                logprobs.append(0.0)

            feature = logprobs
            linear_probs = [math.exp(lp) for lp in feature]

            return response.choices[0].message.content, top_logprobs, linear_probs, feature

        elif self.is_llama:
            loop = asyncio.get_event_loop()

            def llama_generate():
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that answers the user's question using a provided context."},
                    {"role": "user", "content": prompt}
                ]
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    first_token_logits = logits[0, -1, :]
                    log_probs = torch.log_softmax(first_token_logits, dim=-1)

                topk = torch.topk(log_probs, k=5)
                topk_ids = topk.indices.tolist()
                topk_logps = topk.values.tolist()
                topk_tokens = [self.tokenizer.decode([tid]) for tid in topk_ids]

                feature = topk_logps[:5]
                while len(feature) < 5:
                    feature.append(0.0)

                linear_probs = [math.exp(lp) for lp in feature]

                del inputs, input_ids, outputs, logits, first_token_logits, log_probs
                torch.cuda.empty_cache()

                reply = " "  # 如需要可替换为真实生成内容
                return reply, list(zip(topk_tokens, topk_logps)), linear_probs, feature

            return await loop.run_in_executor(None, llama_generate)

        
    async def get_QA_answer_async_defense(self, question: str, cypher_answer: str, strategy: str):
        if strategy == "prompt_modifying":
            prompt = generate_QA_prompt_Prompt_modi(question, cypher_answer)
        elif strategy == "paraphrasing":
            prompt = generate_QA_prompt_Praphra(question, cypher_answer)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if self.is_openai:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers the user's question using a provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                logprobs=True,
                top_logprobs=5
            )

            # 统一处理 top-5 logprobs
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            logprobs = [entry.logprob for entry in top_logprobs[:5]]
            while len(logprobs) < 5:
                logprobs.append(0.0)

            feature = logprobs
            linear_probs = [math.exp(lp) for lp in feature]

            return response.choices[0].message.content, top_logprobs, linear_probs, feature

        elif self.is_llama:
            loop = asyncio.get_event_loop()

            def llama_generate():
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that answers the user's question using a provided context."},
                    {"role": "user", "content": prompt}
                ]
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    first_token_logits = logits[0, -1, :]
                    log_probs = torch.log_softmax(first_token_logits, dim=-1)

                topk = torch.topk(log_probs, k=5)
                topk_ids = topk.indices.tolist()
                topk_logps = topk.values.tolist()
                topk_tokens = [self.tokenizer.decode([tid]) for tid in topk_ids]

                feature = topk_logps[:5]
                while len(feature) < 5:
                    feature.append(0.0)

                linear_probs = [math.exp(lp) for lp in feature]

                del inputs, input_ids, outputs, logits, first_token_logits, log_probs
                torch.cuda.empty_cache()

                reply = " "  # 如果启用 generate 可替换
                return reply, list(zip(topk_tokens, topk_logps)), linear_probs, feature

            return await loop.run_in_executor(None, llama_generate)



