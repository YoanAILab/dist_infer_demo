# dist_infer_demo.py
import os
import time
import torch
import torch.distributed as dist
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def setup_distributed():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def cleanup_distributed():
    dist.destroy_process_group()

def load_model_and_tokenizer(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def infer(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        end = time.time()
    logits = outputs.logits
    pred_token_id = int(torch.argmax(logits[0, -1]))
    pred_token = tokenizer.decode([pred_token_id])
    elapsed_ms = (end - start) * 1000
    return pred_token_id, pred_token.strip(), elapsed_ms

def main():
    rank, world_size = setup_distributed()

    prompts = [
        "Hello world",
        "The sky is",
        "I love",
        "Artificial intelligence is",
        "Python is a popular"
    ]
    prompt = prompts[rank % len(prompts)]

    if rank == 0:
        print(f"🚀 使用 GPT2 Student v2 进行多进程推理（共 {world_size} 个进程）")

    device = torch.device("cuda" if torch.cuda.is_available() and rank == 0 else "cpu")
    model_path = "./gpt2_student_v2"
    tokenizer, model = load_model_and_tokenizer(model_path)
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        end = time.time()

    logits = outputs.logits
    pred_id = int(torch.argmax(logits[0, -1]))
    pred_token = tokenizer.decode([pred_id])
    elapsed = (end - start) * 1000

    time.sleep(rank * 0.1)
    print("\n" + "=" * 60)
    print(f"[Rank {rank}] 📝 Prompt: {prompt}")
    print(f"[Rank {rank}] 🔹 Predicted Token ID: {pred_id}")
    print(f"[Rank {rank}] 🔹 Predicted Token Text: {pred_token.strip()}")
    print(f"[Rank {rank}] ⏱ 推理耗时: {elapsed:.2f} ms")
    print("=" * 60 + "\n")

    cleanup_distributed()

if __name__ == "__main__":
    main()
