import gradio as gr
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
import os

# Log in to Hugging Face with your token
token = os.getenv("HF_TOKEN")  # Replace with your actual token
login(token=token, add_to_git_credential=True)

# Load the model and tokenizer for image analysis
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"

# Load the image analysis model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
tokenizers = AutoTokenizer.from_pretrained(model_id, revision=revision)

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

def process_image(image, symptoms_input):
    # Encode the image
    enc_image = model.encode_image(image)

    # Get the answer about the image
    ans = model.answer_question(enc_image, "Describe what you see in the image such as color, shape, size and what it may be in detail.", tokenizers)

    # Combine answers for the final input
    prompt_question = "Describe what you know about this, including what steps to take and who to talk to about this."
    input_text = ans + " " + prompt_question

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user", "content": input_text}
    ]

    # Apply the chat template
    prompt = tokenizer_text.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)

    # Tokenize the input text and send it to the GPU
    input_ids = tokenizer_text(prompt, return_tensors="pt").to("cuda")

    # Generate outputs from the local LLM
    outputs = llm_model.generate(**input_ids, max_new_tokens=1024)

    # Decode the output tokens to text
    outputs_decoded = tokenizer_text.decode(outputs[0])

    # Clean and format the output
    outputs_cleaned = outputs_decoded.replace("<bos>", "").replace("<eos>", "").replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()

    # Format the output for better readability
    formatted_output = (
        f"### Analysis Result:\n\n"
        f"{outputs_cleaned}\n\n"
        f"---\n"
        f"### Symptoms Provided:\n"
        f"- {symptoms_input}"
    )

    return formatted_output, image  # Return both the formatted output text and the image

# Define the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload an Image",height=300),
        gr.Textbox(lines=2, placeholder="Enter symptoms here...", label="Query")
    ],
    outputs=[
        gr.Textbox(label="Analysis Result", lines=30),  # Text output for analysis
        gr.Image(label="Uploaded Image")                 # Optional: Include image output
    ],
    title="MedAdviser",
    description="Upload an image to analyze irregularities and get medical information. Provide any symptoms you have."
)

# Launch the Gradio app
iface.launch()
