import nest_asyncio
nest_asyncio.apply()
import json
import pandas as pd
import asyncio
import time
from tqdm.asyncio import tqdm_asyncio
from utils.get_feature import TokenProbabilityExtractor
from utils.neo4j_utils import Neo4jQueryRunner
import re
import os

# 初始化组件
encoder = TokenProbabilityExtractor("gpt-4o-mini")
decoder = TokenProbabilityExtractor("gpt-4o-mini")
runner = Neo4jQueryRunner("your_neo4j_uri_here", "your_username_here", "your_password_here")  # 替换为你的密码
semaphore = asyncio.Semaphore(1)

# 核心处理函数
async def process_item(item):
    query = item["question"]
    QA_PROMPT = """
    Answer the question precisely and concisely, using only one word or phrase—no introduction, explanation, prefix, or suffix.
    Question:
        {}
    """.format(query)
    question = QA_PROMPT
    gold_answer = item["answer"]
    # 1. 获取 Cypher 查询
    cypher_response = await encoder.get_cypher_queries_async(question,target_dataset="target_dataset_here")  
    cypher_query = cypher_response.choices[0].message.content.strip()
    cypher_query = re.sub(r"```(?:cypher)?\s*([\s\S]+?)\s*```", r"\1", cypher_query, flags=re.IGNORECASE).strip()
    # 2. 执行 Cypher 查询
    cypher_answer = await runner.run_query_async(cypher_query)

    # 3. 获取 QA 结果与特征
    llm_answer, top_logprobs, linear_probs, feature = await decoder.get_QA_answer_async(query, cypher_answer)

    return {
        "question": query,
        "gold_answer": gold_answer,
        "llm_answer": llm_answer.strip(),
        "next_token_log_prob": top_logprobs,
        "next_token_linear_prob": linear_probs,
        "feature": feature,
        "label": 1 # 1 for member samples, 0 for non-member samples
    }

async def run_with_retry(item, max_retries=3, retry_delay=1.0):
    for attempt in range(max_retries):
        try:
            async with semaphore:
                return await process_item(item)
        except Exception as e:
            print(f"⚠️ No. {attempt + 1} Times of failure: {item['question']}\n Eroor: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))  # 指数退避
            else:
                print(f"❌ Set to default feature after multiple trials：{item['question']}")
                return {
                    "question": item["question"],
                    "gold_answer": item["answer"],
                    "llm_answer": "failed",
                    "next_token_log_prob": [],
                    "next_token_linear_prob": [],
                    "feature": [0.0] * 5,
                    "label": 1   # 0 for non-memeber samples
                }

async def main():
    path = "pos.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # 运行任务带重试
    tasks = [run_with_retry(item) for item in data]
    results = await tqdm_asyncio.gather(*tasks)

    records = [r for r in results if r is not None]

    df = pd.DataFrame(records)

    # ✅ 确保保存路径存在
    output_path = "feature.csv"
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✅ 已保存 CSV 文件：{output_path}")

    await runner.aclose()

if __name__ == "__main__":
    asyncio.run(main())