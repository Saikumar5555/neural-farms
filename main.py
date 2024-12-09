from flask import Flask, request, render_template, jsonify
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from roboflow import Roboflow
from ultralytics import YOLO
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from numpy import asarray

# Initialize Flask app
app = Flask(__name__)

# Load Roboflow model (for object detection)
# rf = Roboflow(api_key="xQfSHHbVWsh0GQ4c6EHP")
# project = rf.workspace("pest-vision-major").project("pest-vision-major-zfqvh")
# version = project.version(6)
# dataset = version.download("yolov11")
model_yolo = YOLO(r"C:\Users\INS1\Documents\final medadviser\runs\detect\train15\weights\best.pt")  # Load YOLOv11 pretrained model for detection

# Load Hugging Face model (for image captioning)
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model_hf = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
tokenizer_hf = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Set up Hugging Face model for text generation
model_id_text = "google/gemma-2b-it"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# Check for flash attention support
if is_flash_attn_2_available() and (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8):
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"  # scaled dot product attention

llm_model = AutoModelForCausalLM.from_pretrained(
    model_id_text,
    torch_dtype=torch.float16,
    attn_implementation=attn_implementation
)

if torch.cuda.is_available():
    llm_model.to("cuda")

# Set the login token (replace with your own Hugging Face token)
token = "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"
login(token=token)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image
    image_file = request.files['image']
    image = Image.open(image_file)

    # Predict using YOLOv5
    results = model_yolo.predict(image)
    detected_objects = []
    for r in results:
        for box in r.boxes:  # Iterate through the detected boxes
            # Get the class ID and map it to the name
            class_id = int(box.cls)  # Get the class index of the detected object
            annotated_image = r.plot()
            object_name = r.names[class_id]  # Get the class name from the 'names' attribute
            print(annotated_image)
            print(object_name)
            
    # Use the first detected object for further processing
    # object_name = detected_objects[0] if detected_objects else "Unknown"
    print(f"Detected object: {object_name}")

    # Generate analysis using Hugging Face model (if supported by your model)
    try:
        image = Image.fromarray(asarray(annotated_image))
        enc_image = model_hf.encode_image(image)
        story_prompt = f"Describe the irregularity in the image such as color, shape, size, and what it may be." + object_name
        ans = model_hf.answer_question(enc_image, story_prompt, tokenizer_hf)
    except Exception as e:
        print(f"Image analysis failed: {e}")
        ans = "Image analysis could not be completed."

    # Generate detailed response based on symptoms and detected object
    symptoms = (
        "Symptoms such as sticky or wet surfaces due to the honeydew they excrete, "
        "which can also promote the growth of sooty mold. The plant may exhibit distorted, curled, or stunted leaves, "
        "and yellowing or chlorosis, especially on new growth. Wilting and dehydration occur as pests feed on the sap, "
        "while a cotton-like white substance can be seen on leaves, stems, or leaf axils."
    )

    patient = (
        "You are a farmer with extensive knowledge of crops, pests, and plant care. "
        "Please answer the following question with practical, field-tested solutions to pest-related problems, "
        "focusing on simple, effective remedies that a farmer would use.\n\n"
        "Patient Symptoms:\n"
        f"{symptoms}\n\n"
        "Question:\n"
        "What is the pest, what are its symptoms, what causes it, how can it be prevented, "
        "and which type of remedies to consider?"
    )

    try:
        # Apply chat template and tokenize input
        prompt = tokenizer_hf.apply_chat_template(
            conversation=[{"role": "user", "content": patient}],
            chat_template='user',
            tokenize=False,
            add_generation_prompt=True
        )
        input_ids = tokenizer_hf(prompt, return_tensors="pt").to("cuda")

        # Generate text response
        outputs = llm_model.generate(**input_ids, max_new_tokens=1024, temperature=0.7, top_k=50)
        outputs_decoded = tokenizer_hf.decode(outputs[0])
    except Exception as e:
        print(f"Text generation failed: {e}")
        outputs_decoded = "Text generation could not be completed."

    # Return response as JSON
    return jsonify({
        'detected_object': object_name,
        'image_analysis': ans,
        'generated_response': outputs_decoded
    })


if __name__ == '__main__':
    # Make Flask app accessible on the local network
    app.run(host='0.0.0.0', port=5000, debug=True)
