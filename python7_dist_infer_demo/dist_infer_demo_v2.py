import os
import torch
import torch.distributed as dist
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

def setup_distributed():
    dist.init_process_group(
        backend="gloo",  # Windows 和 CPU 环境推荐 gloo，Linux GPU 多卡建议用 nccl
        init_method="env://"
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def cleanup():
    dist.destroy_process_group()

def main():
    rank, world_size = setup_distributed()
    torch.manual_seed(42)

    # === 设备分配策略 ===
    if torch.cuda.is_available() and rank == 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # === 加载模型与 tokenizer ===
    model_path = "./gpt2_student_v2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()

    prompts = [
        "Hello world",
        "The sky is",
        "I love",
        "Artificial intelligence is"
    ]

    if rank >= len(prompts):
        print(f"[Rank {rank}] 🚫 无任务，直接退出")
        cleanup()
        return

    prompt = prompts[rank]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        end = time.time()

    logits = outputs.logits
    pred_token_id = int(logits[0, -1].argmax())
    pred_token = tokenizer.decode([pred_token_id]).strip()
    latency_ms = (end - start) * 1000

    result = {
        "rank": rank,
        "prompt": prompt,
        "pred_token_id": pred_token_id,
        "pred_token": pred_token,
        "latency_ms": latency_ms
    }

    # === 进程间通信同步结果 ===
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, result)

    if rank == 0:
        print("\n🚀 所有进程推理结果：")
        for r in sorted(gathered, key=lambda x: x["rank"]):
            print(f"\n[Rank {r['rank']}] 📝 Prompt: {r['prompt']}")
            print(f"[Rank {r['rank']}] 🔹 Predicted Token ID: {r['pred_token_id']}")
            print(f"[Rank {r['rank']}] 🔹 Predicted Token Text: {r['pred_token']}")
            print(f"[Rank {r['rank']}] ⏱ 推理耗时: {r['latency_ms']:.2f} ms")

    cleanup()

if __name__ == "__main__":
    main()
