import os
import sys
import traceback

# 禁用 PyTorch CUDA 内存缓存，避免与fakeGPU内存管理冲突
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
# 禁用 vision 相关依赖，避免 torchvision 假算子注册失败
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ.setdefault("TORCH_SDPA_KERNEL", "math")

try:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 完全禁用 torchvision
    transformers.utils.import_utils._torchvision_available = False
    transformers.utils.import_utils._torchvision_version = "0.0"

    # 确保使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    model_path_base = os.path.expanduser("~/models")
    model_name = os.path.join(model_path_base, "Qwen/Qwen2.5-0.5B-Instruct")

    print(f"\nLoading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    print("Model loaded successfully!")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Say hello"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    print(f"\nGenerating response on {device}...")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=20
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Response:", response)
    print("\n=== TEST PASSED ===")

except Exception as e:
    print(f"\n=== TEST FAILED ===")
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)
