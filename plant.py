import gradio as gr
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
import os
from ultralytics import YOLO
import numpy as np
from roboflow import Roboflow

# Set up environment variables and login
token = os.getenv("access_token")  # Replace with your actual token if needed
if token:
    login(token=token, add_to_git_credential=True)

# Load the text generation model for plant information
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
    attn_implementation=attn_implementation,
    quantization_config=quantization_config
)

# Move model to CUDA if available
if torch.cuda.is_available():
    llm_model.to("cuda")

# Load YOLOv8 model for plant detection
# You can replace this path with your trained model path
yolo_model = YOLO(r"C:\Users\intelliodns\runs\detect\train48\weights\best.pt")

# Optional: Connect to Roboflow for additional functionality
def setup_roboflow():
    rf = Roboflow(api_key="xQfSHHbVWsh0GQ4c6EHP")
    project = rf.workspace("mama-veaxb").project("plant-detection-em72z")
    version = project.version(1)
    return version

# Process the plant image
def process_plant_image(image, query_input):
    # Save the original PIL image temporarily to work with YOLOv8
    temp_img_path = "temp_plant_image.jpg"
    if isinstance(image, Image.Image):
        image.save(temp_img_path)
    else:
        Image.fromarray(image).save(temp_img_path)
    
    # Run YOLOv8 detection on the image with CPU device to avoid CUDA NMS issues
    results = yolo_model(temp_img_path, device="cpu")
    
    # Get the detection visualization
    detected_image = results[0].plot()
    
    # Convert back to PIL image
    detected_image_pil = Image.fromarray(detected_image)
    
    # Clean up temporary file
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    
    # Extract detection information
    detection_info = ""
    for r in results:
        # Extract detected plants and their counts
        if len(r.boxes) > 0:
            classes = r.boxes.cls.cpu().numpy()
            names = r.names
            unique_classes, counts = np.unique(classes, return_counts=True)
            
            detection_info = "Detected plants: "
            for cls, count in zip(unique_classes, counts):
                detection_info += f"{count} {names[int(cls)]}, "
            detection_info = detection_info.rstrip(", ")
            
            # Skip adding confidence scores to keep the output clean
        else:
            detection_info = "No plants detected"
    
    # Create prompt for the LLM model
    prompt_question = f"""Based on the detected plants, provide detailed information about:
1. Plant characteristics and growth stages
2. Optimal growing conditions (soil type, temperature, water requirements)
3. Common diseases and pests that might affect this crop
4. Expected yield and harvesting time
5. Nutritional value and agricultural importance
6. Any sustainable farming practices recommended for this crop

Additional user query: {query_input}"""
    
    # Combine information for the final input
    input_text = f"Plant detection results: {detection_info}. {prompt_question}"

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user", "content": input_text}
    ]

    # Apply the chat template
    prompt = tokenizer_text.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)

    # Tokenize the input text and send it to the GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = tokenizer_text(prompt, return_tensors="pt").to(device)

    # Generate outputs from the local LLM
    outputs = llm_model.generate(**input_ids, max_new_tokens=1024)

    # Decode the output tokens to text
    outputs_decoded = tokenizer_text.decode(outputs[0])

    # Clean and format the output
    outputs_cleaned = outputs_decoded.replace("<bos>", "").replace("<eos>", "")
    outputs_cleaned = outputs_cleaned.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()
    
    # Get the model's response (extract just the assistant's part)
    if "model" in outputs_cleaned.lower():
        parts = outputs_cleaned.split("model:", 1)
        if len(parts) > 1:
            outputs_cleaned = parts[1].strip()

    # Format the output for better readability
    formatted_output = (
        f"### Crop Analysis Result:\n\n"
        f"{outputs_cleaned}\n\n"
        f"---\n"
        f"### Detection Summary:\n"
        f"{detection_info}\n\n"
        f"### User Query:\n"
        f"{query_input}"
    )

    # Return the formatted output text and the detected image
    return formatted_output, detected_image_pil

# Define the Gradio interface
iface = gr.Interface(
    fn=process_plant_image,
    inputs=[
        gr.Image(type="pil", label="Upload Plant Image", height=300),
        gr.Textbox(lines=2, placeholder="Ask about the detected plants...", label="Query")
    ],
    outputs=[
        gr.Textbox(label="Plant Analysis", lines=30),
        gr.Image(label="Detection Results")
    ],
    title="CropAdviser",
    description="Upload a crop/plant image for comprehensive agricultural information. Get detailed analysis on growth stages, optimal conditions, diseases, yield potential, and sustainable farming practices."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()