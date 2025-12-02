# feedback_model.py


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
LORA_PATH = "llama-qlora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Base model (4-bit QLoRA)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load LoRA on top
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# (Optional) Merge weights for inference
try:
    model = model.merge_and_unload()
except:
    pass

model.eval()


def generate_feedback(password: str) -> str:
    prompt = (
        "Analyze the following password.\n"
        f"Password: {password}\n"
        "Explain in a single sentence what exactly to do to improve it, and why it is weak.\n"
        "\nResponse:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.3,
            top_p=0.9,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

