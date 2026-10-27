import argparse
import base64
import json
import time
from openai import OpenAI
import os
import tiktoken
from PIL import Image
import io
from tqdm import tqdm
import math
import concurrent.futures

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "You are a helpful assistant. You are given one or more images of the same scene (possibly from different viewpoints) and a task description. "
    "Multiple narrations may be provided in a single task description, separated by ' / '; these are alternative phrasings of the same task. "
    "You need to ONLY describe the objects relevant to the task. "
    "Formulate your response as a list of objects. Do not list the robot arm or anything attached to it. "
    "Never list objects that cannot be directly manipulated by the robot arm (e.g., table, desk, workbench, floor, wall), "
    "but you may use them to ground the objects (e.g., 'white cloth on the table'). "
    "When applicable, use distinguishing attributes such as color, material, size, or relative position to describe objects and avoid ambiguity "
    "(e.g., 'wooden stand on the table' instead of 'stand'). "
    "Only list the most relevant objects, typically no more than 4. "
    "The listed objects should be directly relevant to the task and usually already mentioned in the task description. "
    "Return None if you cannot find any objects relevant to the task. "
    "Example outputs: ['black marker on the table'], ['left drawer'], ['blue cup', 'white lid'], ['None']"
)

def encode_image(path: str, resize_to: tuple = (512, 512)) -> str:
    """
    Resize and encode image as base64 for OpenAI API.
    
    Args:
        path: Path to the image file
        resize_to: Target size (width, height). Defaults to (512, 512)
    
    Returns:
        Base64 encoded string of the resized image
    """
    # Open and resize image
    with Image.open(path) as img:
        # Convert RGBA to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create a white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img_resized = img.resize(resize_to, Image.Resampling.LANCZOS)
        
        # Encode to base64
        buffer = io.BytesIO()
        img_resized.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")


def get_image_cost(model: str = "gpt-5-nano") -> float:
    """
    Get the fixed cost per 512x512 image for different models.
    
    Args:
        model: The model name
    
    Returns:
        Cost in USD per 512x512 image
    """
    # Fixed cost per 512x512 image (USD)
    IMAGE_COSTS = {
        "gpt-5-nano": 0.000019,
        "gpt-5-mini": 0.000077,
        "gpt-4o-mini": 0.000425,
        "gpt-4o": 0.000213,
    }
    
    return IMAGE_COSTS.get(model, 0.000019)


def calculate_text_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Calculate the number of tokens for a text string.
    
    Args:
        text: The text to tokenize
        model: The model name (for selecting the right tokenizer)
    
    Returns:
        Number of tokens in the text
    """
    # For gpt-4o models, use o200k_base encoding
    if "gpt-4o" in model or "gpt-4-turbo" in model or "gpt-5-nano" in model:
        encoding = tiktoken.get_encoding("o200k_base")
    else:
        encoding = tiktoken.encoding_for_model(model)
    
    return len(encoding.encode(text))


def get_model_pricing(model: str) -> dict:
    """
    Get pricing per 1M tokens for text tokens.
    
    Args:
        model: The model name
    
    Returns:
        Dictionary with 'input' and 'output' prices per 1M tokens (for text only)
    """
    # Prices per 1M tokens (USD) - TEXT TOKENS ONLY
    # Source: https://openai.com/api/pricing/
    MODEL_PRICING = {
        # GPT-5 series
        "gpt-5.2": {"input": 0.875, "output": 7.00},
        "gpt-5.1": {"input": 0.625, "output": 5.00},
        "gpt-5": {"input": 0.625, "output": 5.00},
        "gpt-5-mini": {"input": 0.125, "output": 1.00},
        "gpt-5-nano": {"input": 0.025, "output": 0.20},
        "gpt-5.2-pro": {"input": 10.50, "output": 84.00},
        "gpt-5-pro": {"input": 7.50, "output": 60.00},
        
        # GPT-4 series
        "gpt-4.1": {"input": 1.00, "output": 4.00},
        "gpt-4.1-mini": {"input": 0.20, "output": 0.80},
        "gpt-4.1-nano": {"input": 0.05, "output": 0.20},
        "gpt-4o": {"input": 1.25, "output": 5.00},
        "gpt-4o-2024-05-13": {"input": 2.50, "output": 7.50},
        "gpt-4o-mini": {"input": 0.075, "output": 0.30},
        
        # O series
        "o1": {"input": 7.50, "output": 30.00},
        "o1-pro": {"input": 75.00, "output": 300.00},
        "o3-pro": {"input": 10.00, "output": 40.00},
        "o3": {"input": 1.00, "output": 4.00},
        "o3-deep-research": {"input": 5.00, "output": 20.00},
        "o4-mini": {"input": 0.55, "output": 2.20},
        "o4-mini-deep-research": {"input": 1.00, "output": 4.00},
        "o3-mini": {"input": 0.55, "output": 2.20},
        "o1-mini": {"input": 0.55, "output": 2.20},
        
        # Other
        "computer-use-preview": {"input": 1.50, "output": 6.00},
    }
    
    return MODEL_PRICING.get(model, {"input": 0.025, "output": 0.20})


def calculate_input_tokens(inputs, model: str = "gpt-5-nano", estimated_output_tokens_per_request: int = 50) -> dict:
    """
    Calculate total text tokens and estimated costs for a batch of requests.
    
    Images are assumed to be resized to 512x512 with a fixed cost per image.
    
    Args:
        inputs: List of input dictionaries with 'image_paths' (list) or 'image_path' (str) and 'task' keys
        model: The model name being used
        estimated_output_tokens_per_request: Estimated output tokens per request for cost calculation
    
    Returns:
        Dictionary with token breakdown, total, and pricing information
    """
    total_text_tokens = 0
    total_num_images = 0
    breakdown = []
    
    # Get pricing for this model
    pricing = get_model_pricing(model)
    image_cost_per_image = get_image_cost(model)
    
    # System prompt tokens (counted for each request)
    system_tokens = calculate_text_tokens(SYSTEM_PROMPT, model)
    
    for idx, item in enumerate(inputs):
        # Support both single image_path (string) and multiple image_paths (list)
        if "image_paths" in item:
            image_paths = item["image_paths"] if isinstance(item["image_paths"], list) else [item["image_paths"]]
        elif "image_path" in item:
            image_paths = [item["image_path"]]
        else:
            raise ValueError(f"Item {idx} must have either 'image_path' or 'image_paths' key")
        
        num_images_in_request = len(image_paths)
        total_num_images += num_images_in_request
        
        # Calculate text tokens for this request
        task_tokens = calculate_text_tokens(item["task"], model)
        
        # Total text tokens for this request
        request_text_tokens = system_tokens + task_tokens
        total_text_tokens += request_text_tokens
        
        breakdown.append({
            "index": idx,
            "image_paths": image_paths,
            "num_images": num_images_in_request,
            "system_tokens": system_tokens,
            "task_tokens": task_tokens,
            "text_tokens_total": request_text_tokens,
        })
    
    # Calculate costs
    total_output_tokens = estimated_output_tokens_per_request * len(inputs)
    text_input_cost = (total_text_tokens / 1_000_000) * pricing["input"]
    image_input_cost = total_num_images * image_cost_per_image
    output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
    total_cost = text_input_cost + image_input_cost + output_cost
    
    return {
        "total_text_tokens": total_text_tokens,
        "num_images": total_num_images,
        "estimated_output_tokens": total_output_tokens,
        "num_requests": len(inputs),
        "avg_text_tokens_per_request": total_text_tokens / len(inputs) if inputs else 0,
        "pricing": {
            "model": model,
            "text_input_price_per_1m": pricing["input"],
            "image_cost_per_image": image_cost_per_image,
            "output_price_per_1m": pricing["output"],
            "text_input_cost_usd": text_input_cost,
            "image_input_cost_usd": image_input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost
        },
        "breakdown": breakdown
    }

def get_text_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """
    Get a text embedding for the given text.
    """
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding




def iter_chunks(items, chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size], start


def submit_batches(
    inputs,
    batch_size: int = 220,
    output_dir: str = "object_identification",
    max_workers: int = 8,
    model: str = "gpt-5-nano",
    run_tag: str | None = None,
):
    """
    Submit batch jobs in mini groups and return a list of batch ids.
    Builds and saves requests incrementally to reduce memory usage.
    """
    batch_ids = []
    os.makedirs(output_dir, exist_ok=True)
    chunks = list(iter_chunks(inputs, batch_size))

    def build_batch_file(chunk_info):
        chunk, start_idx, chunk_idx = chunk_info
        if run_tag:
            input_file = os.path.join(output_dir, f"batch_input_{run_tag}_{chunk_idx:04d}.jsonl")
        else:
            input_file = os.path.join(output_dir, f"batch_input_{chunk_idx:04d}.jsonl")
        if os.path.exists(input_file):
            return input_file
        with open(input_file, "w") as f:
            for idx, item in enumerate(chunk, start=start_idx):
                # Support both single image_path (string) and multiple image_paths (list)
                if "image_paths" in item:
                    image_paths = item["image_paths"] if isinstance(item["image_paths"], list) else [item["image_paths"]]
                elif "image_path" in item:
                    image_paths = [item["image_path"]]
                else:
                    raise ValueError(f"Item {idx} must have either 'image_path' or 'image_paths' key")

                # Use episode_id as custom_id if available, otherwise use index
                custom_id = item.get("episode_id", f"sample-{idx}")

                # Build content array starting with task text
                content = [{"type": "text", "text": item["task"]}]

                # Add all images
                for img_path in image_paths:
                    image_b64 = encode_image(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    })

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT
                            },
                            {
                                "role": "user",
                                "content": content,
                            },
                        ],
                        # Enforce List[str] output
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "object_list",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "objects": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "maxItems": 3
                                        }
                                    },
                                    "required": ["objects"],
                                    "additionalProperties": False
                                },
                            },
                        },
                    },
                }

                # Write request immediately
                f.write(json.dumps(request) + "\n")
        return input_file

    if not chunks:
        return batch_ids

    chunk_infos = [(chunk, start_idx, chunk_idx) for chunk_idx, (chunk, start_idx) in enumerate(chunks)]
    worker_count = min(max_workers, len(chunk_infos))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        input_files = list(executor.map(build_batch_file, chunk_infos))

    def submit_single_batch(file_path: str) -> str:
        uploaded = client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch.id

    worker_count = min(max_workers, len(input_files))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        for batch_id in executor.map(submit_single_batch, input_files):
            batch_ids.append(batch_id)

    return batch_ids


def cache_batch_output(outputs: dict, batch_id: str, output_dir: str = "object_identification"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"batch_output_{batch_id}.json")
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    return output_path


def fetch_results(batch_id):
    """
    Poll batch until completion, then return parsed outputs.
    """
    def _find_first_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for v in value.values():
                found = _find_first_list(v)
                if found is not None:
                    return found
        return None

    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        print(f"Batch status: {batch.status}")
        time.sleep(10)

    print(f"Final batch status: {batch.status}")
    print(f"Batch details: {batch}")
    
    if batch.status != "completed":
        raise RuntimeError(f"Batch failed with status {batch.status}")

    output_file_id = batch.output_file_id
    
    # Check if output_file_id is None
    if output_file_id is None:
        print(f"Error file id: {batch.error_file_id}")
        if batch.error_file_id:
            error_content = client.files.content(batch.error_file_id)
            print(f"Error content: {error_content.text}")
        raise RuntimeError(f"Batch completed but output_file_id is None. Request counts: {batch.request_counts}")
    
    content = client.files.content(output_file_id)

    results = {}
    for line in content.text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = record["custom_id"]
        
        # Parse the response - the format depends on whether the request succeeded
        if "response" in record and record["response"]["status_code"] == 200:
            response_body = record["response"]["body"]
            message_content = response_body["choices"][0]["message"]["content"]
            parsed_content = json.loads(message_content)
            try:
                objects = _find_first_list(parsed_content)
                if objects is None:
                    print(f"No list found for {custom_id}: {parsed_content}")
                    results[custom_id] = []
                else:
                    results[custom_id] = objects
            except Exception as exc:
                print(f"Error parsing {custom_id}: {exc}; content: {parsed_content}")
                results[custom_id] = []
        else:
            print(f"Error for {custom_id}: {record.get('error', 'Unknown error')}")
            results[custom_id] = []

    return results

def build_inputs(
    episode_meta_file: str,
    processed_data_path: str,
):
    with open(episode_meta_file, 'r') as f:
        episode_meta = json.load(f)
    inputs = []
    for episode_id, episode_meta_info in tqdm(episode_meta.items(), desc="Building inputs"):
        episode_rel_path = episode_meta_info['rel_path']
        task_description_list = list(episode_meta_info['task_descriptions'].values())
        task_descriptions = " / ".join(task_description_list)
        cam_name_to_serial = episode_meta_info['cam_name_to_serial']
        img_list = []
        for cam_name in ["external_cam1", "external_cam2"]:
            if not os.path.exists(os.path.join(processed_data_path, episode_id, "rgb", cam_name)):
                continue
            image_path = os.path.join(processed_data_path, episode_id, "rgb", cam_name, "0000000000.jpg")
            if not os.path.exists(image_path):
                print(f"warning: image not found for episode {episode_id} and cam {cam_name}")
                continue
            img_list.append(image_path)

        if len(img_list) > 0:
            inputs.append({
                "episode_id": episode_id,
                "image_paths": img_list,
                "task": task_descriptions,
            })
    print(f"Built {len(inputs)} inputs")
    return inputs

if __name__ == "__main__":
    # Configuration
    data_dir = os.environ.get("ROBODREAM_DATA_DIR", "datasets")
    parser = argparse.ArgumentParser(description="Identify task-relevant objects per DROID episode with the OpenAI Batch API.")
    parser.add_argument("--processed_dir", default=os.path.join(data_dir, "processed", "droid_traj2demo", "real"))
    parser.add_argument("--output_json", default=os.path.join(data_dir, "object_segmentation_prompts.json"))
    parser.add_argument("--work_dir", default=os.path.join(data_dir, "object_identification"),
                        help="Where batch request/response jsonl files are cached")
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--yes", action="store_true", help="Submit without the interactive cost confirmation")
    args = parser.parse_args()

    episode_meta_file = os.path.join(args.processed_dir, "filtered_episode_meta25.json")
    output_json_file = args.output_json
    model = args.model

    # Build inputs from episode metadata
    inputs = build_inputs(episode_meta_file=episode_meta_file, processed_data_path=args.processed_dir)

    # If output file exists, rerun only episodes with empty list or "None"
    existing_outputs = {}
    run_tag = None
    if os.path.exists(output_json_file):
        with open(output_json_file, "r") as f:
            existing_outputs = json.load(f)
        rerun_ids = {
            episode_id
            for episode_id, objects in existing_outputs.items()
            if not objects
            or (isinstance(objects, list) and any(obj == "None" or obj is None for obj in objects))
        }
        if rerun_ids:
            inputs = [item for item in inputs if item.get("episode_id") in rerun_ids]
            print(f"Rerun mode: {len(rerun_ids)} episodes need retry; {len(inputs)} inputs found.")
            run_tag = f"rerun_{int(time.time())}"
        else:
            print("Rerun mode: no episodes need retry. Exiting.")
            exit()
    
    # Calculate input tokens and pricing before submitting
    token_info = calculate_input_tokens(inputs, model=model, estimated_output_tokens_per_request=50)
    
    print(f"\n{'='*70}")
    print(f"TOKEN USAGE & COST ESTIMATION")
    print(f"{'='*70}")
    print(f"Model: {token_info['pricing']['model']}")
    print(f"Number of requests: {token_info['num_requests']}")
    print(f"Total images: {token_info['num_images']} (512x512 each)")
    print(f"\nText Tokens:")
    print(f"  Input tokens:  {token_info['total_text_tokens']:,}")
    print(f"  Output tokens: {token_info['estimated_output_tokens']:,} (estimated)")
    print(f"  Avg text tokens per request: {token_info['avg_text_tokens_per_request']:.1f}")
    print(f"\nPricing:")
    print(f"  Text input:  ${token_info['pricing']['text_input_price_per_1m']:.3f} per 1M tokens")
    print(f"  Image input: ${token_info['pricing']['image_cost_per_image']:.6f} per 512x512 image")
    print(f"  Text output: ${token_info['pricing']['output_price_per_1m']:.3f} per 1M tokens")
    print(f"\nEstimated Costs:")
    print(f"  Text input:  ${token_info['pricing']['text_input_cost_usd']:.6f}")
    print(f"  Image input: ${token_info['pricing']['image_input_cost_usd']:.6f} ({token_info['num_images']} images)")
    print(f"  Text output: ${token_info['pricing']['output_cost_usd']:.6f}")
    print(f"  {'─'*40}")
    print(f"  Total cost:  ${token_info['pricing']['total_cost_usd']:.6f}")
    # print(f"\nRequest Breakdown:")
    # for item in token_info['breakdown']:
    #     print(f"  Request {item['index']}: {item['num_images']} image(s), {item['text_tokens_total']:,} text tokens "
    #           f"(system: {item['system_tokens']}, task: {item['task_tokens']})")
    print(f"{'='*70}\n")
    
    if not args.yes and input("Submit batch job? (y/n): ") != "y":
        exit()

    # Submit batch jobs
    batch_ids = submit_batches(
        inputs,
        batch_size=200,
        output_dir=os.path.join(args.work_dir, "batch_inputs"),
        model=model,
        max_workers=16,
        run_tag=run_tag,
    )
    print("\nSubmitted batches:", batch_ids)

    # Fetch results (episode_id -> list of objects)
    outputs = dict(existing_outputs)
    for batch_id in tqdm(batch_ids):
        batch_outputs = fetch_results(batch_id)
        cache_batch_output(batch_outputs, batch_id, output_dir=os.path.join(args.work_dir, "batch_outputs"))
        outputs.update(batch_outputs)
    
    # Save results to JSON file
    with open(output_json_file, 'w') as f:
        json.dump(outputs, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_json_file}")
    print(f"  Total episodes processed: {len(outputs)}")
    print(f"\nSample outputs:")
    for i, (episode_id, objects) in enumerate(outputs.items()):
        if i >= 5:  # Show first 5 examples
            break
        print(f"  {episode_id}: {objects}")
    
    if len(outputs) > 5:
        print(f"  ... and {len(outputs) - 5} more")
