AI Classroom Assistant: Visual Question Answering with OpenVINO



This project demonstrates the power of Intel's OpenVINO toolkit to accelerate a Visual Question Answering (VQA) model. The application uses a pre-trained ViLT model (dandelin/vilt-b32-finetuned-vqa) to answer natural language questions about an image.
The interactive web interface, built with Streamlit, allows users to see firsthand the impact of model optimization. You can seamlessly switch between different hardware devices (like CPU and integrated GPU) and compare the performance (latency and model size) of the original PyTorch model against its optimized OpenVINO versions (FP32, FP16, and INT8).



Features
Visual Question Answering (VQA): Upload any image, ask a question, and get an AI-generated answer.
Interactive Web UI: A user-friendly interface built with Streamlit.
Dynamic Hardware Selection: Leverage OpenVINO to run inference on different hardware devices available on your system (e.g., CPU, iGPU).
Real-time Performance Comparison: Instantly compare the performance of different model precisions:
Original PyTorch (FP32)
OpenVINO (FP32)
OpenVINO (FP16 - half-precision)
OpenVINO (INT8 - 8-bit quantized)
Live Metrics: View the inference latency (in milliseconds) and model size (in megabytes) for each configuration.



How It Works
The project follows a robust two-stage process to ensure a fast and responsive user experience.
Stage 1: One-Time Model Preparation (1_prepare_models.py)
This is a setup script that you only need to run once. It performs all the heavy lifting upfront:
Downloads the base ViLT model and its processor from the Hugging Face Hub.
Converts the PyTorch model to the OpenVINO Intermediate Representation (IR) format (FP32).
Compresses the FP32 model to create a half-precision FP16 version, which is smaller and faster.
Quantizes the FP32 model using a calibration dataset and the NNCF framework to create an ultra-fast, low-precision INT8 version.
These optimized models are saved in the optimized_model/ directory.
Stage 2: The Streamlit Application (app.py)
This script runs the interactive web application. It loads the pre-optimized models from Stage 1, ensuring the UI is lightweight and starts quickly. When you select a hardware device and model precision in the app, the OpenVINO Runtime efficiently compiles the chosen model for that specific target and runs the inference.


File Descriptions
requirements.txt: Lists all Python dependencies required for the project.
1_prepare_models.py: A one-time setup script to download the base PyTorch model and convert/optimize it into three OpenVINO formats (FP32, FP16, and INT8).
app.py: The main Streamlit web application that provides the user interface for running inference and comparing the models.
README.md: This file, providing documentation for the project.
































































