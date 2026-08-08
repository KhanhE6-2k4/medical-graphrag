import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import tiktoken

load_dotenv()


SYSTEM_PROMPT = """
---Role---
You are a helpful assistant responding to user queries.

---Goal---
Generate direct and concise answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know".

---Knowledge Base---
{context_data}
"""


def format_context(contexts: Any) -> str:
    if contexts is None:
        return ""

    if isinstance(contexts, str):
        return contexts.strip()

    if isinstance(contexts, list):
        chunks = []

        for i, ctx in enumerate(contexts):
            if isinstance(ctx, str):
                text = ctx.strip()
            elif isinstance(ctx, dict):
                text = (
                    ctx.get("text")
                    or ctx.get("content")
                    or ctx.get("chunk")
                    or ctx.get("context")
                    or json.dumps(ctx, ensure_ascii=False)
                )
                text = str(text).strip()
            else:
                text = str(ctx).strip()

            if text:
                chunks.append(f"[Chunk {i + 1}]\n{text}")

        return "\n\n".join(chunks)

    return str(contexts).strip()


def get_contexts_from_item(item: Dict[str, Any]) -> Any:
    if "contexts" in item:
        return item["contexts"]
    if "context" in item:
        return item["context"]
    if "chunk_context" in item:
        return item["chunk_context"]
    return []


def build_client(mode: str, base_url: Optional[str]) -> OpenAI:
    if mode.upper() != "API":
        raise ValueError(
            f"Unsupported mode: {mode}. This script currently supports only --mode API."
        )

    api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not set. Set it first:\n"
            ".env:\n"
            "  LLM_API_KEY=your_api_key\n"
            "Linux/WSL/macOS:\n"
            "  export LLM_API_KEY='your_api_key'\n"
            "Windows PowerShell:\n"
            "  $env:LLM_API_KEY='your_api_key'"
        )

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI(api_key=api_key)

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Hàm phụ trợ tính chính xác số token của một đoạn văn bản"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def generate_answer(
    client: OpenAI,
    model: str,
    question: str,
    contexts: Any,
    max_output_tokens: int = 256,
    max_retries: int = 3,
) -> str:
    context_data = format_context(contexts)

    # Cắt gọn context nếu vượt quá 20,000 ký tự
    max_context_chars = 20000 
    if len(context_data) > max_context_chars:
        context_data = context_data[:max_context_chars]

    instructions = SYSTEM_PROMPT.format(context_data=context_data)

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=question,
                temperature=0,
                max_output_tokens=max_output_tokens,
            )
            return response.output_text.strip()

        except Exception as e:
            if attempt == max_retries - 1:
                # return f"API_ERROR: {repr(e)}"
                raise RuntimeError(f"API_ERROR: Kết nối thất bại sau {max_retries} lần thử. Chi tiết: {repr(e)}")   

            time.sleep(2 ** attempt)

    return "API_ERROR: unknown error"


def normalize_input_data(data: Any) -> List[Dict[str, Any]]:
    """
    1. Flat list format:
       [
         {
           "id": "...",
           "question_type": "...",
           "question": "...",
           "contexts": [...]
         }
       ]

    2. Grouped format:
       {
         "Fact Retrieval": {
           "average_scores": {...},
           "detailed": [...]
         },
         "Complex Reasoning": {
           "average_scores": {...},
           "detailed": [...]
         }
       }
    """
    samples = []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        for question_type, block in data.items():
            if isinstance(block, dict) and isinstance(block.get("detailed"), list):
                for item in block["detailed"]:
                    if isinstance(item, dict):
                        copied = dict(item)
                        copied.setdefault("question_type", question_type)
                        samples.append(copied)

            elif isinstance(block, list):
                for item in block:
                    if isinstance(item, dict):
                        copied = dict(item)
                        copied.setdefault("question_type", question_type)
                        samples.append(copied)

    return samples


def filter_samples(
    samples: List[Dict[str, Any]],
    question_type_filter: Optional[List[str]] = None,
    limit_per_type: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if question_type_filter:
        allowed_types = set(question_type_filter)
        samples = [
            item for item in samples
            if item.get("question_type") in allowed_types
        ]

    if limit_per_type is not None:
        type_counts = {}
        limited_samples = []

        for item in samples:
            qtype = item.get("question_type", "UNKNOWN")
            current_count = type_counts.get(qtype, 0)

            if current_count < limit_per_type:
                limited_samples.append(item)
                type_counts[qtype] = current_count + 1

        samples = limited_samples

    if limit is not None:
        samples = samples[:limit]

    return samples

def process_item(
    item,
    client,
    model,
    max_output_tokens,
    detailed_output,
):
    question = item.get("question", "")
    contexts = get_contexts_from_item(item)

    if not question:
        generated_answer = "API_ERROR: missing question"
    else:
        generated_answer = generate_answer(
            client=client,
            model=model,
            question=question,
            contexts=contexts,
            max_output_tokens=max_output_tokens,
        )

    result_item = {
        "id": item.get("id"),
        "source": item.get("source"),
        "question_type": item.get("question_type"),
        "question": question,
        "generated_answer": generated_answer,
    }

    if "ground_truth" in item:
        result_item["ground_truth"] = item["ground_truth"]
    elif "answer" in item:
        result_item["ground_truth"] = item["answer"]

    for key in [
        "evidence",
        "context_relevancy",
        "evidence_recall",
    ]:
        if key in item:
            result_item[key] = item[key]

    if detailed_output:
        result_item["context"] = contexts

    return result_item


def run_answer_generation(
    mode: str,
    model: str,
    base_url: Optional[str],
    data_file: str,
    output_file: str,
    detailed_output: bool = False,
    max_output_tokens: int = 256,
    limit: Optional[int] = None,
    question_type_filter: Optional[List[str]] = None,
    limit_per_type: Optional[int] = None,
    max_workers: int = 10,
) -> List[Dict[str, Any]]:
    input_path = Path(data_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = normalize_input_data(data)

    samples = filter_samples(
        samples=samples,
        question_type_filter=question_type_filter,
        limit_per_type=limit_per_type,
        limit=limit,
    )

    print(f"Loaded {len(samples)} samples for generation.")

    type_counter = Counter(item.get("question_type", "UNKNOWN") for item in samples)
    print("Selected samples by question type:")
    for qtype, count in type_counter.items():
        print(f"  {qtype}: {count}")

    if question_type_filter:
        print(f"Question type filter: {question_type_filter}")

    if limit_per_type is not None:
        print(f"Limit per type: {limit_per_type}")

    print(f"Max output tokens: {max_output_tokens}")

    client = build_client(mode=mode, base_url=base_url)

    #---------------------------------------------
    results = []

    worker = partial(
        process_item,
        client=client,
        model=model,
        max_output_tokens=max_output_tokens,
        detailed_output=detailed_output,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Gửi tất cả các tác vụ vào hàng đợi luồng
        futures = {executor.submit(worker, sample): sample for sample in samples}
        
        # Tạo thanh tiến trình
        progress_bar = tqdm(total=len(samples), desc="Generating answers")
        
        try:
            for future in as_completed(futures):
                try:
                    # Lấy kết quả từ luồng, nếu luồng đó dính API_ERROR -> ném ngoại lệ 
                    result_item = future.result()
                    results.append(result_item)
                    progress_bar.update(1)
                except Exception as e:
                    # Đóng thanh tiến trình và in thông báo lỗi hệ thống
                    progress_bar.close()
                    print(f"\n[HỆ THỐNG DỪNG LẠI CHỦ ĐỘNG] Phát hiện lỗi nghiêm trọng trong luồng xử lý!")
                    print(f"Chi tiết lỗi: {e}")
                    
                    # Hủy toàn bộ các tác vụ khác đang nằm trong hàng đợi chưa kịp chạy
                    executor.shutdown(wait=False, cancel_futures=True)
                    
                    # Thoát ngay lập tức với mã lỗi 1
                    os._exit(1)
        finally:
            progress_bar.close()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} generated answers to: {output_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate answers from retrieved contexts using an OpenAI-compatible API."
    )

    parser.add_argument("--mode", type=str, required=True, choices=["API"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    parser.add_argument("--detailed_output", action="store_true")
    parser.add_argument("--max_output_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--question_type",
        nargs="+",
        default=None,
        help=(
            "Only generate answers for selected question types. "
            "Example: --question_type 'Fact Retrieval' 'Complex Reasoning'"
        ),
    )

    parser.add_argument(
        "--limit_per_type",
        type=int,
        default=None,
        help="Limit number of samples per selected question type.",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Number of concurrent API requests."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_answer_generation(
        mode=args.mode,
        model=args.model,
        base_url=args.base_url,
        data_file=args.data_file,
        output_file=args.output_file,
        detailed_output=args.detailed_output,
        max_output_tokens=args.max_output_tokens,
        limit=args.limit,
        question_type_filter=args.question_type,
        limit_per_type=args.limit_per_type,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()