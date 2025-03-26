from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig

# Log in to Hugging Face with your token
token = "HF_TOKEN"  # Replace with your actual token
login(token=token)

# Load the model and tokenizer for image analysis
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"

# Load the image analysis model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
tokenizers = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Load the image
image_path = r'C:\inttelliod projectts\source\Screenshot 2024-10-12 173026.png'
image = Image.open(image_path)

# Encode the image
enc_image = model.encode_image(image)

# Get the answer about the image
ans = model.answer_question(enc_image, "Describe the irregularity in the image such as color, shape, size and what it may be.", tokenizers)

# Print the image analysis answer
print(ans)

# Define symptoms and patient inquiry
symptoms = ("Crusting of skin bumps. Cysts. Papules (small red bumps). "
            "Pustules (small red bumps containing white or yellow pus). "
            "Redness around the skin eruptions. Scarring of the skin. "
            "Whiteheads. Blackheads.")
patient = ("What is the disease, what are its symptoms, what causes it, "
           "how can it be prevented, and which type of doctor to consider?")

# Combine answers for the final input
input_text = ans
print(f"Input text:\n{input_text}")

# Configure model for text generation
model_id_text = "google/gemma-2b-it"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# Check for flash attention support
if is_flash_attn_2_available() and (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8):
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"  # scaled dot product attention

# Load the text model and tokenizer
tokenizer_text = AutoTokenizer.from_pretrained(model_id_text)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id_text,
    torch_dtype=torch.float16,
    attn_implementation=attn_implementation
)

# Move model to CUDA if available
if torch.cuda.is_available():
    llm_model.to("cuda")

# Create prompt template for instruction-tuned model
dialogue_template = [
    {"role": "user", "content": input_text}
]

# Apply the chat template
prompt = tokenizer_text.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
print(f"\nPrompt (formatted):\n{prompt}")

# Tokenize the input text (turn it into numbers) and send it to the GPU
input_ids = tokenizer_text(prompt, return_tensors="pt").to("cuda")

# Generate outputs from the local LLM
outputs = llm_model.generate(**input_ids, max_new_tokens=1024)

# Decode the output tokens to text
outputs_decoded = tokenizer_text.decode(outputs[0])
print(f"Model output (decoded):\n{outputs_decoded}\n")
