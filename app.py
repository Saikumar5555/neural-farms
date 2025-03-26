# import gradio as gr
# from huggingface_hub import login
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from PIL import Image
# import torch
# from transformers.utils import is_flash_attn_2_available
# from transformers import BitsAndBytesConfig
# import os
# import torch
# print(torch.cuda.is_available())  # This should print True if CUDA is enabled

# # Log in to Hugging Face with your token
# token = os.getenv("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")  # Replace with your actual token
# login(token=token, add_to_git_credential=True)

# # Load the model and tokenizer for image analysis
# model_id = "vikhyatk/moondream2"
# revision = "2024-08-26"

# # Load the image analysis model and tokenizer
# model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
# tokenizers = AutoTokenizer.from_pretrained(model_id, revision=revision)

# # Configure model for text generation
# model_id_text = "google/gemma-2b-it"
# quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# # Check for flash attention support
# if is_flash_attn_2_available() and (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8):
#     attn_implementation = "flash_attention_2"
# else:
#     attn_implementation = "sdpa"  # scaled dot product attention

# # Load the text model and tokenizer
# tokenizer_text = AutoTokenizer.from_pretrained(model_id_text)
# llm_model = AutoModelForCausalLM.from_pretrained(
#     model_id_text,
#     torch_dtype=torch.float16,
#     attn_implementation=attn_implementation
# )

# # Move model to CUDA if available
# if torch.cuda.is_available():
#     llm_model.to("cuda")




# def process_image(image, symptoms_input):
#     # Encode the image
#     enc_image = model.encode_image(image)

#     # Get the answer about the image
#     ans = model.answer_question(enc_image, "Describe what you see in the image such as color, shape, size and what it may be in detail.", tokenizers)

#     # Combine answers for the final input
#     prompt_question = "Describe what you know about this, including what steps to take and who to talk to about this."
#     input_text = ans + " " + prompt_question

#     # Create prompt template for instruction-tuned model
#     dialogue_template = [
#         {"role": "user", "content": input_text}
#     ]

#     # Apply the chat template
#     prompt = tokenizer_text.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)

#     # Tokenize the input text and send it to the GPU
#     input_ids = tokenizer_text(prompt, return_tensors="pt").to("cuda")

#     # Generate outputs from the local LLM
#     outputs = llm_model.generate(**input_ids, max_new_tokens=1024)

#     # Decode the output tokens to text
#     outputs_decoded = tokenizer_text.decode(outputs[0])

#     # Clean and format the output
#     outputs_cleaned = outputs_decoded.replace("<bos>", "").replace("<eos>", "").replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()

#     # Format the output for better readability
#     formatted_output = (
#         f"### Analysis Result:\n\n"
#         f"{outputs_cleaned}\n\n"
#         f"---\n"
#         f"### Symptoms Provided:\n"
#         f"- {symptoms_input}"
#     )

#     return formatted_output, image  # Return both the formatted output text and the image

# # Define the Gradio interface
# iface = gr.Interface(
#     fn=process_image,
#     inputs=[
#         gr.Image(type="pil", label="Upload an Image",height=300),
#         gr.Textbox(lines=2, placeholder="Enter symptoms here...", label="Query")
#     ],
#     outputs=[
#         gr.Textbox(label="Analysis Result", lines=30),  # Text output for analysis
#         gr.Image(label="Uploaded Image")                 # Optional: Include image output
#     ],
#     title="MedAdviser",
#     description="Upload an image to analyze irregularities and get medical information. Provide any symptoms you have."
# )

# # Launch the Gradio app
# iface.launch()

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////

# import gradio as gr
# from huggingface_hub import login
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from PIL import Image
# import torch
# from transformers.utils import is_flash_attn_2_available
# from transformers import BitsAndBytesConfig
# import os

# # Log in to Hugging Face with your token
# token = os.getenv("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")  # Replace with your actual token
# login(token=token, add_to_git_credential=True)

# # Load the model and tokenizer for image analysis
# model_id = "vikhyatk/moondream2"
# revision = "2024-08-26"

# # Load the image analysis model and tokenizer
# model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
# tokenizers = AutoTokenizer.from_pretrained(model_id, revision=revision)

# # Configure model for text generation
# model_id_text = "google/gemma-2b-it"
# quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# # Check for flash attention support
# if is_flash_attn_2_available() and (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8):
#     attn_implementation = "flash_attention_2"
# else:
#     attn_implementation = "sdpa"  # scaled dot product attention

# # Load the text model and tokenizer
# tokenizer_text = AutoTokenizer.from_pretrained(model_id_text)
# llm_model = AutoModelForCausalLM.from_pretrained(
#     model_id_text,
#     torch_dtype=torch.float16,
#     attn_implementation=attn_implementation
# )

# # Move model to CUDA if available
# if torch.cuda.is_available():
#     llm_model.to("cuda")

# def process_image(image, symptoms_input):
#     # Example: If you're using a different image encoder, make sure you adjust this part
#     # Here, you would need a proper method to encode the image before passing it to the model
#     # You can use feature extractors or pre-trained image processing models.
    
#     # For simplicity, let's assume the model can handle PIL images
#     enc_image = model.encode_image(image)  # Ensure this is a valid method

#     # Get the answer about the image
#     ans = model.answer_question(enc_image, "Describe what you see in the image such as color, shape, size and what it may be in detail.", tokenizers)

#     # Combine answers for the final input
#     prompt_question = "Describe what you know about this, including what steps to take and who to talk to about this."
#     input_text = ans + " " + prompt_question

#     # Create prompt template for instruction-tuned model
#     dialogue_template = [
#         {"role": "user", "content": input_text}
#     ]

#     # Apply the chat template
#     prompt = tokenizer_text.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)

#     # Tokenize the input text and send it to the GPU
#     input_ids = tokenizer_text(prompt, return_tensors="pt").to("cuda")

#     # Generate outputs from the local LLM
#     outputs = llm_model.generate(**input_ids, max_new_tokens=1024)

#     # Decode the output tokens to text
#     outputs_decoded = tokenizer_text.decode(outputs[0])

#     # Clean and format the output
#     outputs_cleaned = outputs_decoded.replace("<bos>", "").replace("<eos>", "").replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()

#     # Format the output for better readability
#     formatted_output = (
#         f"### Analysis Result:\n\n"
#         f"{outputs_cleaned}\n\n"
#         f"---\n"
#         f"### Symptoms Provided:\n"
#         f"- {symptoms_input}"
#     )

#     # Return both the formatted output text and the image
#     return formatted_output, image  # Gradio will automatically handle displaying the image

# # Define the Gradio interface
# iface = gr.Interface(
#     fn=process_image,
#     inputs=[
#         gr.Image(type="pil", label="Upload an Image", height=300),
#         gr.Textbox(lines=2, placeholder="Enter symptoms here...", label="Query")
#     ],
#     outputs=[
#         gr.Textbox(label="Analysis Result", lines=30),  # Text output for analysis
#         gr.Image(label="Uploaded Image")  # Image output will display the uploaded image
#     ],
#     title="MedAdviser",
#     description="Upload an image to analyze irregularities and get medical information. Provide any symptoms you have."
# )

# # Launch the Gradio app
# iface.launch()





import gradio as gr
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
import os
from ultralytics import YOLO  # Import for YOLOv8
import numpy as np

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

# Load YOLOv8 model with CPU inference to avoid CUDA NMS issue
yolo_model = YOLO(r"C:\Saikumar\Neural farms\model\best.pt")  # Replace with your model path 

def process_image(image, symptoms_input):
    # Save the original PIL image temporarily to work with YOLOv8
    temp_img_path = "temp_image.jpg"
    if isinstance(image, Image.Image):
        image.save(temp_img_path)
    else:
        Image.fromarray(image).save(temp_img_path)
    
    # Run YOLOv8 detection on the image with CPU device to avoid CUDA NMS issues
    results = yolo_model(temp_img_path, device="cpu")
    
    # Get the detection visualization (plot returns numpy array)
    detected_image = results[0].plot()
    
    # Convert back to PIL image
    detected_image_pil = Image.fromarray(detected_image)
    
    # Clean up temporary file
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    
    # Get moondream2 to analyze the original image
    # Convert back to PIL if needed
    original_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    enc_image = model.encode_image(original_image)
    
    # Get the answer about the image
    ans = model.answer_question(enc_image, "Describe what you see in the image such as color, shape, size and what it may be in detail.", tokenizers)
    
    # Get YOLO detection results for text analysis
    detection_info = ""
    for r in results:
        # Extract detected objects and their counts
        if len(r.boxes) > 0:
            classes = r.boxes.cls.cpu().numpy()
            names = r.names
            unique_classes, counts = np.unique(classes, return_counts=True)
            
            detection_info = "Detected: "
            for cls, count in zip(unique_classes, counts):
                detection_info += f"{count} {names[int(cls)]}, "
            detection_info = detection_info.rstrip(", ")
        else:
            detection_info = "No objects detected"
    
    # Combine answers for the final input
    prompt_question = "Describe what you know about this, including what steps to take and who to talk to about this."
    input_text = ans + " " + detection_info + " " + prompt_question

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
        f"### YOLO Detection Results:\n"
        f"{detection_info}\n\n"
        f"### Symptoms Provided:\n"
        f"- {symptoms_input}"
    )

    # Return the formatted output text and the detected image
    return formatted_output, detected_image_pil

# Define the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload an Image", height=300),
        gr.Textbox(lines=2, placeholder="Enter symptoms here...", label="Query")
    ],
    outputs=[
        gr.Textbox(label="Analysis Result", lines=30),  # Text output for analysis
        gr.Image(label="Detection Results")  # Changed label to indicate it shows detection results
    ],
    title="MedAdviser",
    description="Upload an image to analyze irregularities and get medical information. Provides YOLOv8 detection results and medical analysis."
)

# Launch the Gradio app
iface.launch()




#  pest_model_path =r"C:\Saikumar\Neural farms\model\best.pt"  # Update with your path
#     plant_model_path =  r"C:\Users\intelliodns\runs\detect\train48\weights\best.pt"  # Update with your path