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
import tempfile
from typing import Tuple, List, Dict  # Added import for type hints

# Set up environment variables and login
token = os.getenv("ACCESS_TOKEN")  # Replace with your actual token if needed
if token:
    login(token=token, add_to_git_credential=True)

# Configuration for text generation model
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

# Load both detection models
pest_model = YOLO(r"C:\Users\intelliodns\runs\detect\train47\weights\best.pt")  # Path to pest detection model
plant_model = YOLO(r"C:\Users\intelliodns\runs\detect\train48\weights\best.pt")  # Path to plant detection model

def load_image(file_path: str) -> Image.Image:
    """Load an image from file path"""
    return Image.open(file_path).convert("RGB")


def analyze_image(image: Image.Image, model_type: str) -> Tuple[str, Image.Image, List[str]]:
    """
    Analyze an image using either pest or plant detection model.
    
    Args:
        image: PIL Image to analyze
        model_type: Either 'pest' or 'plant'
        
    Returns:
        Tuple of (detection_info, detected_image, detected_classes)
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_path = temp_file.name
        image.save(temp_path)
    
    # Select the appropriate model
    model = pest_model if model_type == "pest" else plant_model
    
    # Run YOLOv8 detection on the image with CPU device to avoid CUDA NMS issues
    results = model(temp_path, device="cpu")
    
    # Get the detection visualization
    detected_image = results[0].plot()
    
    # Convert back to PIL image
    detected_image_pil = Image.fromarray(detected_image)
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # Extract detection information and class names
    detection_info = ""
    detected_classes = []
    for r in results:
        # Extract detected objects and their counts
        if len(r.boxes) > 0:
            classes = r.boxes.cls.cpu().numpy()
            names = r.names
            unique_classes, counts = np.unique(classes, return_counts=True)
            
            detection_info = f"Detected {model_type}s: "
            for cls, count in zip(unique_classes, counts):
                class_name = names[int(cls)]
                detection_info += f"{count} {class_name}, "
                detected_classes.append(class_name)
            detection_info = detection_info.rstrip(", ")
        else:
            detection_info = f"No {model_type}s detected"
    
    return detection_info, detected_image_pil, list(set(detected_classes))  # Remove duplicates

def generate_pest_analysis(detected_pests: List[str], query: str) -> str:
    """
    Generate detailed pest-specific analysis using the LLM.
    
    Args:
        detected_pests: List of detected pest names
        query: User's additional question
        
    Returns:
        Formatted analysis text
    """
    if not detected_pests:
        return "No pests detected in the image."
    
    pests_list = ", ".join(detected_pests)
    prompt_question = f"""For the detected pests ({pests_list}), provide detailed information about:
1. Scientific name and classification
2. Life cycle and behavior
3. Damage symptoms to look for
4. Prevention methods (cultural, biological)
5. Treatment options (organic and chemical)
6. Economic threshold levels
7. Recommended pesticides (with safety precautions)
8. Best application methods and timing

Additional user query: {query}"""
    
    # Create prompt template
    dialogue_template = [
        {"role": "user", "content": prompt_question}
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
    
    # Get the model's response
    if "model" in outputs_cleaned.lower():
        parts = outputs_cleaned.split("model:", 1)
        if len(parts) > 1:
            outputs_cleaned = parts[1].strip()

    # Format the output for better readability
    formatted_output = (
        f"### Pest Analysis for {pests_list}:\n\n"
        f"{outputs_cleaned}\n\n"
        f"### User Query:\n"
        f"{query}"
    )

    return formatted_output

def generate_plant_analysis(detected_plants: List[str], query: str) -> str:
    """
    Generate detailed plant-specific analysis using the LLM.
    
    Args:
        detected_plants: List of detected plant names
        query: User's additional question
        
    Returns:
        Formatted analysis text
    """
    if not detected_plants:
        return "No plants detected in the image."
    
    plants_list = ", ".join(detected_plants)
    prompt_question = f"""For the detected plants ({plants_list}), provide detailed information about:
1. Scientific name and classification
2. Growth stages and characteristics
3. Optimal growing conditions
4. Common pests and diseases
5. Nutritional requirements
6. Harvesting and post-harvest handling
7. Economic importance
8. Sustainable cultivation practices

Additional user query: {query}"""
    
    # Create prompt template
    dialogue_template = [
        {"role": "user", "content": prompt_question}
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
    
    # Get the model's response
    if "model" in outputs_cleaned.lower():
        parts = outputs_cleaned.split("model:", 1)
        if len(parts) > 1:
            outputs_cleaned = parts[1].strip()

    # Format the output for better readability
    formatted_output = (
        f"### Plant Analysis for {plants_list}:\n\n"
        f"{outputs_cleaned}\n\n"
        f"### User Query:\n"
        f"{query}"
    )

    return formatted_output

def process_images(files: List[str], pest_query: str, plant_query: str) -> Dict:
    """
    Process multiple images, detecting both pests and plants.
    """
    results = {
        "pest_analysis": [],
        "plant_analysis": [],
        "pest_images": [],
        "plant_images": []
    }
    
    if not files:
        return results
    
    for file_path in files:
        try:
            img = load_image(file_path)
            
            # Process pests if query provided
            if pest_query.strip():
                pest_info, pest_img, pest_classes = analyze_image(img, "pest")
                pest_analysis = generate_pest_analysis(pest_classes, pest_query)
                results["pest_analysis"].append(f"{pest_info}\n\n{pest_analysis}")
                results["pest_images"].append(pest_img)
            
            # Process plants if query provided
            if plant_query.strip():
                plant_info, plant_img, plant_classes = analyze_image(img, "plant")
                plant_analysis = generate_plant_analysis(plant_classes, plant_query)
                results["plant_analysis"].append(f"{plant_info}\n\n{plant_analysis}")
                results["plant_images"].append(plant_img)
                
        except Exception as e:
            print(f"Error processing image {file_path}: {str(e)}")
            continue
    
    return results

def main_interface(files: List[str], pest_query: str, plant_query: str):
    """
    Main function for the Gradio interface.
    """
    if not files:
        return "Please upload images", [], "Please upload images", []
    
    results = process_images(files, pest_query, plant_query)
    
    pest_output = "\n\n---\n\n".join(results["pest_analysis"]) if results["pest_analysis"] else "No pest analysis performed"
    plant_output = "\n\n---\n\n".join(results["plant_analysis"]) if results["plant_analysis"] else "No plant analysis performed"
    
    return pest_output, results["pest_images"], plant_output, results["plant_images"]

# Define the Gradio interface
with gr.Blocks(title="AgroVision Analyzer") as demo:
    gr.Markdown("""
    # AgroVision Analyzer
    Upload images to detect and analyze both pests and plants
    """)
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(file_count="multiple", file_types=["image"], label="Upload Images")
            pest_query = gr.Textbox(label="Pest-related Questions", 
                                 placeholder="Ask about pest identification, damage symptoms, or control measures...")
            plant_query = gr.Textbox(label="Plant-related Questions",
                                   placeholder="Ask about plant identification, growth requirements, or cultivation practices...")
            submit_btn = gr.Button("Analyze")
        
        with gr.Column():
            pest_output = gr.Textbox(label="Pest Analysis Results")
            pest_gallery = gr.Gallery(label="Pest Detections")
            
        with gr.Column():
            plant_output = gr.Textbox(label="Plant Analysis Results")
            plant_gallery = gr.Gallery(label="Plant Detections")
    
    submit_btn.click(
        fn=main_interface,
        inputs=[file_input, pest_query, plant_query],
        outputs=[pest_output, pest_gallery, plant_output, plant_gallery]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()