from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
import os
from huggingface_hub import login

# Initialize FastAPI app
app = FastAPI()

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

# Function to process the image and symptoms input
def process_image(image: Image.Image, symptoms_input: str):
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

    # Manually create the formatted prompt
    prompt = '\n'.join([f"{message['role']}: {message['content']}" for message in dialogue_template])

    # Tokenize and generate the response
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

    return formatted_output

# FastAPI endpoint to handle image and symptoms input
@app.post("/analyze/")
async def analyze_image(image: UploadFile = File(...), symptoms: str = Form(...)):
    # Convert the uploaded image to a PIL Image
    image_data = await image.read()
    pil_image = Image.open(BytesIO(image_data))

    # Process the image and symptoms input
    result = process_image(pil_image, symptoms)

    # Return the result
    return {"result": result}

# Simple HTML form for testing (you can ignore this for API use)
@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <html>
        <body>
            <h2>Upload Image and Enter Symptoms</h2>
            <form action="/analyze/" method="post" enctype="multipart/form-data">
                <input type="file" name="image" required><br><br>
                <textarea name="symptoms" placeholder="Enter symptoms here..." rows="4" cols="50" required></textarea><br><br>
                <input type="submit" value="Analyze">
            </form>
        </body>
    </html>
    """

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
