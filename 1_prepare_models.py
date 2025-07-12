# 1_prepare_models.py

import os
# Set the environment variable BEFORE importing transformers to prevent warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import openvino
import nncf
import torch

# --- Configuration ---
# =======================================================================
# THE FIX: Switched to a smaller model to reduce memory usage.
MODEL_ID = "microsoft/DialoGPT-small"
# =======================================================================
OPTIMIZED_MODEL_DIR = Path("optimized_model_chatbot")
FP32_PATH = OPTIMIZED_MODEL_DIR / "dialogpt_fp32.xml"
FP16_PATH = OPTIMIZED_MODEL_DIR / "dialogpt_fp16.xml"
INT8_PATH = OPTIMIZED_MODEL_DIR / "dialogpt_int8.xml"

def main():
    """
    This is a one-time script to download, convert, and optimize all models.
    """
    if INT8_PATH.exists() and FP16_PATH.exists() and FP32_PATH.exists():
        print("✅ All chatbot models are already prepared. You can now run the Streamlit app.")
        return

    print("--- Starting One-Time Chatbot Model Preparation ---")
    OPTIMIZED_MODEL_DIR.mkdir(exist_ok=True)
    
    print(f"➡️ [1/5] Downloading base model '{MODEL_ID}' from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pytorch_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_cache=False)
    print("✅ [1/5] Download complete.")
    
    # Create a dummy input for model conversion
    dummy_input_text = "Hello, how are you?"
    inputs = tokenizer(dummy_input_text, return_tensors="pt")
    example_input = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask}

    print("➡️ [2/5] Converting PyTorch model to OpenVINO FP32...")
    ov_model_fp32 = openvino.convert_model(pytorch_model, example_input=example_input)
    openvino.save_model(ov_model_fp32, str(FP32_PATH))
    print("✅ [2/5] OpenVINO FP32 model saved.")

    print("➡️ [3/5] Compressing model to OpenVINO FP16...")
    openvino.save_model(ov_model_fp32, str(FP16_PATH), compress_to_fp16=True)
    print("✅ [3/5] OpenVINO FP16 model saved.")

    print("➡️ [4/5] Preparing calibration dataset for INT8 quantization...")
    
    def get_calibration_data(num_samples=20):
        calibration_texts = [
            "What is OpenVINO?", "Tell me about yourself.", "What can you do?",
            "Can you write a poem?", "Explain what a neural network is.", "Who are you?",
            "Hello, nice to meet you.", "Let's talk about technology.", "What is the weather like today?",
            "Do you have any hobbies?", "What is the capital of France?", "Can you help me with my homework?",
            "Tell me a joke.", "I'm feeling happy today.", "What's the meaning of life?",
            "How does a computer work?", "Let's chat for a bit.", "Can you recommend a good book?",
            "What are your thoughts on AI?", "Goodbye!"
        ]
        for i in range(num_samples):
            text = calibration_texts[i % len(calibration_texts)]
            inputs = tokenizer(text, return_tensors="pt")
            yield {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask}

    calibration_dataset = nncf.Dataset(get_calibration_data())
    
    print("➡️ [5/5] Quantizing model to OpenVINO INT8 (this may take a few minutes)...")
    
    int8_model = nncf.quantize(
        ov_model_fp32, 
        calibration_dataset, 
        preset=nncf.QuantizationPreset.PERFORMANCE,
        subset_size=10
    )
    
    openvino.save_model(int8_model, str(INT8_PATH))
    print("✅ [5/5] OpenVINO INT8 model saved.")
    
    print("\n🎉 --- All chatbot models have been prepared successfully! --- 🎉")
    print("You can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()

