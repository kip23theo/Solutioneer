# app.py

import streamlit as st
import time
import openvino
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
# =======================================================================
# THE FIX: Switched to the smaller model to match the preparation script.
MODEL_ID = "microsoft/DialoGPT-small"
# =======================================================================
OPTIMIZED_MODEL_DIR = Path("optimized_model_chatbot")
FP32_PATH = OPTIMIZED_MODEL_DIR / "dialogpt_fp32.xml"
FP16_PATH = OPTIMIZED_MODEL_DIR / "dialogpt_fp16.xml"
INT8_PATH = OPTIMIZED_MODEL_DIR / "dialogpt_int8.xml"

# --- Page Configuration ---
st.set_page_config(page_title="Classroom Assistant with OpenVINO", page_icon="🤖", layout="wide")

# --- Caching & Model Loading ---
@st.cache_resource
def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)

@st.cache_resource
def get_pytorch_model():
    return AutoModelForCausalLM.from_pretrained(MODEL_ID)

@st.cache_resource
def get_all_openvino_models(device):
    """Loads and compiles all OpenVINO models for a specific device."""
    core = openvino.Core()
    models = {
        "OpenVINO (FP32)": core.compile_model(str(FP32_PATH), device),
        "OpenVINO (FP16)": core.compile_model(str(FP16_PATH), device),
        "OpenVINO (INT8)": core.compile_model(str(INT8_PATH), device),
    }
    return models

def get_model_size(model_type):
    path = None
    if model_type == "Original PyTorch (FP32)":
        # =======================================================================
        # THE FIX: Updated hardcoded size for the smaller model.
        return 351.49
        # =======================================================================
    elif model_type == "OpenVINO (FP32)": path = FP32_PATH
    elif model_type == "OpenVINO (FP16)": path = FP16_PATH
    elif model_type == "OpenVINO (INT8)": path = INT8_PATH
    
    if path and path.exists():
        return os.path.getsize(path.with_suffix('.bin')) / (1024 * 1024)
    return 0

# --- Main Application ---
st.title("🤖 Chatbot Assistant (Optimized with OpenVINO™)")

if not all([FP32_PATH.exists(), FP16_PATH.exists(), INT8_PATH.exists()]):
    st.error("Models not found! Please run the setup script from your terminal first:")
    st.code("python 1_prepare_models.py")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.title("Controls")
core = openvino.Core()
available_devices = core.available_devices
if "CPU" not in available_devices: available_devices.insert(0, "CPU")
device_choice = st.sidebar.selectbox("1. Choose Hardware Device:", available_devices)

try:
    openvino_models = get_all_openvino_models(device_choice)
except Exception as e:
    st.sidebar.error(f"Failed to load models for {device_choice}. It may not be supported. Error: {e}")
    st.stop()

model_choice = st.sidebar.selectbox(
    "2. Choose Model Precision:",
    ["Original PyTorch (FP32)", "OpenVINO (FP32)", "OpenVINO (FP16)", "OpenVINO (INT8)"]
)
st.sidebar.markdown("---")

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you today?"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
            tokenizer = get_tokenizer()
            
            chat_history_ids = torch.tensor([])
            for i, message in enumerate(st.session_state.messages):
                if i < len(st.session_state.messages) -1:
                    input_ids = tokenizer.encode(message["content"] + tokenizer.eos_token, return_tensors='pt')
                    chat_history_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if chat_history_ids.numel() > 0 else input_ids

            new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

            start_time = time.perf_counter()
            
            if model_choice == "Original PyTorch (FP32)":
                if device_choice != "CPU": st.warning("PyTorch model will run on CPU only.", icon="⚠️")
                model = get_pytorch_model()
                chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
                full_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            else: # OpenVINO inference
                compiled_model = openvino_models[model_choice]
                
                output_tokens = []
                for _ in range(100): # Max generation length
                    inputs = {"input_ids": bot_input_ids, "attention_mask": torch.ones_like(bot_input_ids)}
                    result = compiled_model(inputs)
                    
                    next_token_logits = torch.from_numpy(result[compiled_model.output(0)][:, -1, :])
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    
                    if next_token_id == tokenizer.eos_token_id:
                        break
                    
                    output_tokens.append(next_token_id.item())
                    bot_input_ids = torch.cat([bot_input_ids, next_token_id], dim=-1)
                
                full_response = tokenizer.decode(output_tokens)
                
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000
            model_size = get_model_size(model_choice)
            
            message_placeholder.markdown(full_response)
            
            st.sidebar.markdown("---")
            st.sidebar.header("Last Turn Performance")
            st.sidebar.metric("Inference Time", f"{latency:.2f} ms")
            st.sidebar.metric("Model Size", f"{model_size:.2f} MB")

    st.session_state.messages.append({"role": "assistant", "content": full_response})

with st.expander("How this works: The OpenVINO Optimization Pipeline"):
    st.markdown("""
    This app demonstrates how OpenVINO can optimize a **conversational AI model** (`microsoft/DialoGPT-small`).
    """)
