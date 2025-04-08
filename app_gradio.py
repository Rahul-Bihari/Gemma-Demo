"""
Gemma Demo Application

An interactive demo for Gemma, Google's state-of-the-art lightweight open models,
using Gradio for the interface and Hugging Face Hub for model access.
This version provides a more graceful fallback when full dependencies aren't available.
"""

import os
import logging
import sys
import random
import time
import json
from datetime import datetime

# Import example prompts and utilities
from examples import (
    TEXT_GENERATION_EXAMPLES,
    QUESTION_ANSWERING_EXAMPLES,
    CODE_GENERATION_EXAMPLES,
    CREATIVE_WRITING_EXAMPLES
)
from utils import format_generated_text, format_code_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Try to import Gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    logger.warning("Gradio not available. Some features will be limited.")
    GRADIO_AVAILABLE = False

# Try to import Torch and Transformers
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers and Torch available - using real model integration")
except ImportError:
    logger.warning("Torch or Transformers not available. Using alternative API integration.")
    TRANSFORMERS_AVAILABLE = False

# Try to import the Hugging Face API client as an alternative
try:
    from huggingface_api import HuggingFaceAPIClient
    HUGGINGFACE_API_AVAILABLE = True
    logger.info("Hugging Face API client available - can use API for model interaction")
except ImportError:
    logger.warning("Hugging Face API client not available - will use simulated responses")
    HUGGINGFACE_API_AVAILABLE = False

# Simulated version of GemmaDemo for when transformers is not available
class SimulatedGemmaDemo:
    def __init__(self):
        """Initialize the SimulatedGemmaDemo instance."""
        self.model_name = None
        logger.info("Initializing SimulatedGemmaDemo")
        self.examples = {
            "text": TEXT_GENERATION_EXAMPLES,
            "qa": QUESTION_ANSWERING_EXAMPLES,
            "code": CODE_GENERATION_EXAMPLES,
            "creative": CREATIVE_WRITING_EXAMPLES
        }
        
    def load_model(self, model_name):
        """Simulate loading a model"""
        logger.info(f"Simulating loading model: {model_name}")
        self.model_name = model_name
        # Check for Hugging Face token
        hf_token = os.environ.get("HUGGING_FACE_TOKEN")
        if not hf_token:
            return {
                "status": "error", 
                "message": "⚠️ Hugging Face token not found. Please set the HUGGING_FACE_TOKEN environment variable."
            }
        
        if random.random() < 0.2:  # Sometimes simulate an error for realism
            return {
                "status": "error", 
                "message": "⚠️ Simulated error loading model. This is a simulation as transformers package is not available."
            }
        
        # Add small delay to simulate loading
        time.sleep(1)
        return {
            "status": "success", 
            "message": f"Model {model_name} simulated. NOTE: Using simulated responses as transformers package is not available."
        }
        
    def generate_text(self, prompt, max_length=256, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1):
        """Simulate text generation"""
        logger.info(f"Simulating text generation for prompt: {prompt[:50]}...")
        
        if not self.model_name:
            return {"status": "error", "message": "Model not loaded. Please load a model first."}
        
        # Add small delay to simulate processing
        time.sleep(0.5 + random.random() * 2)
        
        # Get a random example as a "simulated" response or modify based on prompt
        example = random.choice(self.examples["text"])
        simulated_response = f"This is a simulated response as transformers package is not available.\n\n{example}"
        
        return {
            "status": "success",
            "generated_text": simulated_response,
            "model_name": self.model_name,
            "simulated": True
        }
    
    def answer_question(self, question, max_length=150, temperature=0.4):
        """Simulate answering a question"""
        logger.info(f"Simulating answering question: {question[:50]}...")
        
        if not self.model_name:
            return {"status": "error", "message": "Model not loaded. Please load a model first."}
        
        # Add small delay to simulate processing
        time.sleep(0.5 + random.random() * 1.5)
        
        # Get a random example as a "simulated" response
        example = random.choice(self.examples["qa"])
        simulated_answer = f"This is a simulated response as transformers package is not available.\n\n{example}"
        
        return {
            "status": "success",
            "answer": simulated_answer,
            "model_name": self.model_name,
            "simulated": True
        }
    
    def generate_code(self, prompt, max_length=256, temperature=0.2):
        """Simulate code generation"""
        logger.info(f"Simulating code generation for prompt: {prompt[:50]}...")
        
        if not self.model_name:
            return {"status": "error", "message": "Model not loaded. Please load a model first."}
        
        # Add small delay to simulate processing
        time.sleep(0.5 + random.random() * 2)
        
        # Get a random example as a "simulated" response
        example = random.choice(self.examples["code"])
        simulated_code = f"# This is a simulated response as transformers package is not available.\n\n```python\n{example}\n```"
        
        return {
            "status": "success",
            "code": format_code_output(simulated_code),
            "model_name": self.model_name,
            "simulated": True
        }
    
    def creative_writing(self, prompt, max_length=400, temperature=1.0):
        """Simulate creative writing"""
        logger.info(f"Simulating creative writing for prompt: {prompt[:50]}...")
        
        if not self.model_name:
            return {"status": "error", "message": "Model not loaded. Please load a model first."}
        
        # Add small delay to simulate processing
        time.sleep(1 + random.random() * 3)
        
        # Get a random example as a "simulated" response
        example = random.choice(self.examples["creative"])
        simulated_text = f"This is a simulated response as transformers package is not available.\n\n{example}"
        
        return {
            "status": "success",
            "text": simulated_text,
            "model_name": self.model_name,
            "simulated": True
        }

# Real implementation of GemmaDemo for when transformers is available
# Class implementation using HuggingFaceAPI for when transformers isn't available but API is
class APIGemmaDemo:
    def __init__(self):
        """Initialize the APIGemmaDemo instance."""
        self.hf_client = None
        self.model_name = None
        self.loaded = False
        self.using_real_model = False
        
        # Check if we can use the HuggingFace API client
        if HUGGINGFACE_API_AVAILABLE:
            self.hf_client = HuggingFaceAPIClient()
            if self.hf_client.token:
                logger.info("Hugging Face token found - will use actual Gemma model via API")
                self.using_real_model = True
            else:
                logger.warning("Hugging Face token not found - using simulated responses")
        else:
            logger.warning("HuggingFace API client not available - using simulated responses")
            
        logger.info("Initialized APIGemmaDemo")
    
    def load_model(self, model_name):
        """Load the Gemma model via Hugging Face API"""
        
        # If we can use the real model via API, try to do so
        if self.using_real_model and self.hf_client:
            try:
                logger.info(f"Setting model via API: {model_name}")
                result = self.hf_client.load_model(model_name)
                
                if "error" in result:
                    logger.error(f"Error loading model via API: {result['error']}")
                    logger.info("Falling back to simulated responses")
                    self.using_real_model = False
                else:
                    self.model_name = model_name
                    self.loaded = True
                    logger.info(f"Successfully configured API for {model_name}")
                    return {
                        "status": "success",
                        "message": f"Successfully configured API for {model_name}",
                        "using_real_model": True
                    }
            
            except Exception as e:
                logger.error(f"Error setting up model via API: {str(e)}")
                logger.info("Falling back to simulated responses")
                self.using_real_model = False
                # Fall through to simulated response
        
        # Simulate loading a model
        self.model_name = model_name
        logger.info(f"Simulating model load for {model_name}")
        return {
            "status": "success",
            "message": f"Model {model_name} simulated.",
            "using_real_model": False
        }
    
    def generate_text(self, prompt, max_length=256, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1):
        """Generate text based on parameters"""
        logger.info(f"Generating text for prompt: {prompt[:50]}...")
        
        # If we have the real model loaded via API, use it
        if self.using_real_model and self.hf_client and self.loaded:
            try:
                # Use the API client to generate text
                result = self.hf_client.generate_text(
                    prompt=prompt,
                    max_length=int(max_length),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty)
                )
                
                # Check for errors
                if "error" in result:
                    logger.error(f"Error from API: {result['error']}")
                    logger.info("Falling back to simulated response")
                else:
                    # Format and clean up the response
                    formatted_text = format_generated_text(result["text"])
                    
                    return {
                        "status": "success",
                        "generated_text": formatted_text,
                        "model_name": self.model_name,
                        "source": "actual_model"
                    }
            except Exception as e:
                logger.error(f"Error using API for text generation: {str(e)}")
                logger.info("Falling back to simulated response")
        
        # If we're here, either the model isn't loaded or an error occurred
        # Fall back to the simulated implementation
        simulated_demo = SimulatedGemmaDemo()
        simulated_demo.model_name = self.model_name
        result = simulated_demo.generate_text(prompt, max_length, temperature, top_p, top_k, repetition_penalty)
        
        # Add source indicator
        result["source"] = "simulated"
        return result
    
    def answer_question(self, question, max_length=150, temperature=0.4):
        """Answer a question using the model"""
        logger.info(f"Answering question: {question[:50]}...")
        
        # If we have the real model loaded via API, use it
        if self.using_real_model and self.hf_client and self.loaded:
            try:
                # Use the API client to answer the question
                result = self.hf_client.answer_question(
                    question=question,
                    max_length=int(max_length),
                    temperature=float(temperature)
                )
                
                # Check for errors
                if "error" in result:
                    logger.error(f"Error from API: {result['error']}")
                    logger.info("Falling back to simulated response")
                else:
                    # Format and clean up the response
                    formatted_answer = format_generated_text(result["text"])
                    
                    return {
                        "status": "success",
                        "answer": formatted_answer,
                        "model_name": self.model_name,
                        "source": "actual_model"
                    }
            except Exception as e:
                logger.error(f"Error using API for question answering: {str(e)}")
                logger.info("Falling back to simulated response")
        
        # If we're here, either the model isn't loaded or an error occurred
        # Fall back to the simulated implementation
        simulated_demo = SimulatedGemmaDemo()
        simulated_demo.model_name = self.model_name
        result = simulated_demo.answer_question(question, max_length, temperature)
        
        # Add source indicator
        result["source"] = "simulated"
        return result
    
    def generate_code(self, prompt, max_length=256, temperature=0.2):
        """Generate code based on the given prompt"""
        logger.info(f"Generating code for prompt: {prompt[:50]}...")
        
        # If we have the real model loaded via API, use it
        if self.using_real_model and self.hf_client and self.loaded:
            try:
                # Use the API client to generate code
                result = self.hf_client.generate_code(
                    prompt=prompt,
                    max_length=int(max_length),
                    temperature=float(temperature)
                )
                
                # Check for errors
                if "error" in result:
                    logger.error(f"Error from API: {result['error']}")
                    logger.info("Falling back to simulated response")
                else:
                    # Format and clean up the response
                    formatted_code = format_code_output(result["text"])
                    
                    return {
                        "status": "success",
                        "code": formatted_code,
                        "model_name": self.model_name,
                        "source": "actual_model"
                    }
            except Exception as e:
                logger.error(f"Error using API for code generation: {str(e)}")
                logger.info("Falling back to simulated response")
        
        # If we're here, either the model isn't loaded or an error occurred
        # Fall back to the simulated implementation
        simulated_demo = SimulatedGemmaDemo()
        simulated_demo.model_name = self.model_name
        result = simulated_demo.generate_code(prompt, max_length, temperature)
        
        # Add source indicator
        result["source"] = "simulated"
        return result
    
    def creative_writing(self, prompt, max_length=400, temperature=1.0):
        """Generate creative writing based on the prompt"""
        logger.info(f"Generating creative writing for prompt: {prompt[:50]}...")
        
        # If we have the real model loaded via API, use it
        if self.using_real_model and self.hf_client and self.loaded:
            try:
                # Use the API client to generate creative writing
                result = self.hf_client.creative_writing(
                    prompt=prompt,
                    max_length=int(max_length),
                    temperature=float(temperature)
                )
                
                # Check for errors
                if "error" in result:
                    logger.error(f"Error from API: {result['error']}")
                    logger.info("Falling back to simulated response")
                else:
                    # Format and clean up the response
                    formatted_text = format_generated_text(result["text"])
                    
                    return {
                        "status": "success",
                        "text": formatted_text,
                        "model_name": self.model_name,
                        "source": "actual_model"
                    }
            except Exception as e:
                logger.error(f"Error using API for creative writing: {str(e)}")
                logger.info("Falling back to simulated response")
        
        # If we're here, either the model isn't loaded or an error occurred
        # Fall back to the simulated implementation
        simulated_demo = SimulatedGemmaDemo()
        simulated_demo.model_name = self.model_name
        result = simulated_demo.creative_writing(prompt, max_length, temperature)
        
        # Add source indicator
        result["source"] = "simulated"
        return result

# Real implementation of GemmaDemo for when transformers is available
class GemmaDemo:
    def __init__(self):
        """Initialize the GemmaDemo instance."""
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = None
        logger.info("Initializing GemmaDemo")

    def load_model(self, model_name):
        """Load a Gemma model from Hugging Face Hub"""
        try:
            logger.info(f"Loading model: {model_name}")
            # Check for Hugging Face token
            hf_token = os.environ.get("HUGGING_FACE_TOKEN")
            if not hf_token:
                return {"status": "error", "message": "Hugging Face token not found. Please set the HUGGING_FACE_TOKEN environment variable."}
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                f"google/{model_name}",
                token=hf_token
            )
            
            # Load model with appropriate quantization for memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                f"google/{model_name}",
                token=hf_token,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Create pipeline
            self.pipeline = TextGenerationPipeline(
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            self.model_name = model_name
            logger.info(f"Model {model_name} loaded successfully")
            return {"status": "success", "message": f"Model {model_name} loaded successfully"}
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return {"status": "error", "message": f"Error loading model: {str(e)}"}

    def generate_text(self, prompt, max_length=256, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1):
        """Generate text using the loaded Gemma model"""
        try:
            if not self.model or not self.tokenizer:
                return {"status": "error", "message": "Model not loaded. Please load a model first."}
            
            logger.info(f"Generating text for prompt: {prompt[:50]}...")
            
            # Prepare inputs
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate with parameters
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=int(max_length),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                do_sample=True,
                num_return_sequences=1,
            )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Format and clean up the response
            # Remove the prompt from the response if it's included
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                
            formatted_response = format_generated_text(generated_text)
            
            return {
                "status": "success",
                "generated_text": formatted_response,
                "model_name": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return {"status": "error", "message": f"Error generating text: {str(e)}"}

    def answer_question(self, question, max_length=150, temperature=0.4):
        """Answer a question using the model"""
        try:
            if not self.model or not self.tokenizer:
                return {"status": "error", "message": "Model not loaded. Please load a model first."}
            
            logger.info(f"Answering question: {question[:50]}...")
            
            # Format the prompt for question answering
            qa_prompt = f"Question: {question}\nAnswer:"
            
            # Generate answer with parameters appropriate for Q&A
            if qa_prompt.lower().strip():
                inputs = self.tokenizer(qa_prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=int(max_length) + len(inputs.input_ids[0]),
                    temperature=float(temperature),
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    do_sample=True,
                    num_return_sequences=1,
                )
                
                # Decode the generated text
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from the response
                if answer.startswith(qa_prompt):
                    answer = answer[len(qa_prompt):].strip()
                
                formatted_answer = format_generated_text(answer)
                
                return {
                    "status": "success",
                    "answer": formatted_answer,
                    "model_name": self.model_name
                }
            else:
                return {"status": "error", "message": "Question cannot be empty"}
                
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {"status": "error", "message": f"Error answering question: {str(e)}"}

    def generate_code(self, prompt, max_length=256, temperature=0.2):
        """Generate code based on the given prompt"""
        try:
            if not self.model or not self.tokenizer:
                return {"status": "error", "message": "Model not loaded. Please load a model first."}
            
            logger.info(f"Generating code for prompt: {prompt[:50]}...")
            
            # Format the prompt for code generation
            code_prompt = f"Write code: {prompt}\n```"
            
            # Generate code with parameters appropriate for code generation (lower temperature)
            if code_prompt.lower().strip():
                inputs = self.tokenizer(code_prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=int(max_length) + len(inputs.input_ids[0]),
                    temperature=float(temperature),
                    top_p=0.95,
                    top_k=60,
                    repetition_penalty=1.2,
                    do_sample=True,
                    num_return_sequences=1,
                )
                
                # Decode the generated text
                code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from the response
                if code.startswith(code_prompt):
                    code = code[len(code_prompt):].strip()
                
                # Ensure the code is properly formatted
                formatted_code = format_code_output(code)
                
                return {
                    "status": "success",
                    "code": formatted_code,
                    "model_name": self.model_name
                }
            else:
                return {"status": "error", "message": "Prompt cannot be empty"}
                
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return {"status": "error", "message": f"Error generating code: {str(e)}"}
        
    def creative_writing(self, prompt, max_length=400, temperature=1.0):
        """Generate creative writing based on the prompt"""
        try:
            if not self.model or not self.tokenizer:
                return {"status": "error", "message": "Model not loaded. Please load a model first."}
            
            logger.info(f"Generating creative writing for prompt: {prompt[:50]}...")
            
            # Format the prompt for creative writing
            creative_prompt = f"Write a creative story about: {prompt}\n\n"
            
            # Generate creative text with parameters appropriate for creative writing (higher temperature)
            if creative_prompt.lower().strip():
                inputs = self.tokenizer(creative_prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=int(max_length) + len(inputs.input_ids[0]),
                    temperature=float(temperature),
                    top_p=0.92,
                    top_k=70,
                    repetition_penalty=1.1,
                    do_sample=True,
                    num_return_sequences=1,
                )
                
                # Decode the generated text
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from the response
                if text.startswith(creative_prompt):
                    text = text[len(creative_prompt):].strip()
                
                formatted_text = format_generated_text(text)
                
                return {
                    "status": "success",
                    "text": formatted_text,
                    "model_name": self.model_name
                }
            else:
                return {"status": "error", "message": "Prompt cannot be empty"}
                
        except Exception as e:
            logger.error(f"Error generating creative writing: {str(e)}")
            return {"status": "error", "message": f"Error generating creative writing: {str(e)}"}

def create_gemma_demo():
    """Create the Gradio interface for the Gemma demo"""
    # Initialize the appropriate demo backend based on available dependencies
    if TRANSFORMERS_AVAILABLE:
        logger.info("Using real model implementation with transformers")
        demo_backend = GemmaDemo()
    elif HUGGINGFACE_API_AVAILABLE:
        logger.info("Using API implementation for model interaction")
        demo_backend = APIGemmaDemo()
    else:
        logger.info("Using simulated implementation - no real model available")
        demo_backend = SimulatedGemmaDemo()
    
    # Define handler functions that connect the UI to the backend
    def load_model_handler(model_display_name):
        model_mapping = {
            "Gemma 2B": "gemma-2b",
            "Gemma 7B": "gemma-7b"
        }
        model_id = model_mapping.get(model_display_name, "gemma-7b")
        result = demo_backend.load_model(model_id)
        if result["status"] == "success":
            return f"✅ {result['message']}"
        else:
            return f"❌ {result['message']}"
    
    def text_generation_handler(prompt, max_length, temperature, top_p, top_k, repetition_penalty):
        if not prompt.strip():
            return "Error: Prompt cannot be empty."
        result = demo_backend.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        if result["status"] == "success":
            return result["generated_text"]
        else:
            return f"Error: {result['message']}"
    
    def qa_handler(question, max_length, temperature):
        if not question.strip():
            return "Error: Question cannot be empty."
        result = demo_backend.answer_question(
            question=question,
            max_length=max_length,
            temperature=temperature
        )
        if result["status"] == "success":
            return result["answer"]
        else:
            return f"Error: {result['message']}"
    
    def code_generation_handler(prompt, max_length, temperature):
        if not prompt.strip():
            return "Error: Prompt cannot be empty."
        result = demo_backend.generate_code(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        if result["status"] == "success":
            return result["code"]
        else:
            return f"Error: {result['message']}"
    
    def creative_writing_handler(prompt, max_length, temperature):
        if not prompt.strip():
            return "Error: Prompt cannot be empty."
        result = demo_backend.creative_writing(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        if result["status"] == "success":
            return result["text"]
        else:
            return f"Error: {result['message']}"
    
    # Create Gradio interface with tabs
    with gr.Blocks(title="Gemma Demo", css="footer {visibility: hidden}") as demo:
        gr.Markdown(
            """
            # 🌟 Gemma Interactive Demo
            
            Explore the capabilities of Google's state-of-the-art lightweight open model.
            """
        )
        
        # Model Selection
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Selection")
                model_dropdown = gr.Dropdown(
                    ["Gemma 2B", "Gemma 7B"], 
                    label="Select Model", 
                    value="Gemma 7B",
                    interactive=True
                )
                load_button = gr.Button("Load Model", variant="primary")
                model_status = gr.Markdown("Model status: Not loaded")
        
        # Create tabs for different capabilities
        with gr.Tabs():
            # Text Generation Tab
            with gr.TabItem("Text Generation"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=4
                        )
                        
                        with gr.Row():
                            text_max_length = gr.Slider(
                                label="Max Length",
                                minimum=32,
                                maximum=512,
                                value=256,
                                step=16
                            )
                            text_temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.1,
                                maximum=1.5,
                                value=0.7,
                                step=0.1
                            )
                        
                        with gr.Accordion("Advanced Options", open=False):
                            with gr.Row():
                                text_top_p = gr.Slider(
                                    label="Top P",
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.9,
                                    step=0.1
                                )
                                text_top_k = gr.Slider(
                                    label="Top K",
                                    minimum=1,
                                    maximum=100,
                                    value=50,
                                    step=1
                                )
                                text_repetition_penalty = gr.Slider(
                                    label="Repetition Penalty",
                                    minimum=1.0,
                                    maximum=2.0,
                                    value=1.1,
                                    step=0.1
                                )
                        
                        text_generate_btn = gr.Button("Generate Text", variant="primary")
                    
                    # Examples Panel
                    with gr.Column(scale=1):
                        gr.Markdown("### Examples")
                        text_examples = gr.Dataset(
                            components=[text_prompt],
                            samples=[[ex["prompt"]] for ex in TEXT_GENERATION_EXAMPLES],
                            headers=["Prompt"],
                            label="Click an example to load it"
                        )
                
                text_output = gr.Textbox(
                    label="Generated Text",
                    placeholder="Generated text will appear here...",
                    lines=8
                )
                
                # Connect the components
                text_generate_btn.click(
                    text_generation_handler,
                    inputs=[text_prompt, text_max_length, text_temperature, text_top_p, text_top_k, text_repetition_penalty],
                    outputs=text_output
                )
            
            # Question Answering Tab
            with gr.TabItem("Question Answering"):
                with gr.Row():
                    with gr.Column(scale=3):
                        question_input = gr.Textbox(
                            label="Question",
                            placeholder="Ask a question...",
                            lines=3
                        )
                        
                        with gr.Row():
                            qa_max_length = gr.Slider(
                                label="Max Answer Length",
                                minimum=32,
                                maximum=256,
                                value=150,
                                step=8
                            )
                            qa_temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.4,
                                step=0.1
                            )
                        
                        qa_btn = gr.Button("Answer Question", variant="primary")
                    
                    # Examples Panel
                    with gr.Column(scale=1):
                        gr.Markdown("### Example Questions")
                        qa_examples = gr.Dataset(
                            components=[question_input],
                            samples=[[ex["question"]] for ex in QUESTION_ANSWERING_EXAMPLES],
                            headers=["Question"],
                            label="Click an example to load it"
                        )
                
                qa_output = gr.Textbox(
                    label="Answer",
                    placeholder="Answer will appear here...",
                    lines=6
                )
                
                # Connect the components
                qa_btn.click(
                    qa_handler,
                    inputs=[question_input, qa_max_length, qa_temperature],
                    outputs=qa_output
                )
            
            # Code Generation Tab
            with gr.TabItem("Code Generation"):
                with gr.Row():
                    with gr.Column(scale=3):
                        code_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the code you want...",
                            lines=3
                        )
                        
                        with gr.Row():
                            code_max_length = gr.Slider(
                                label="Max Length",
                                minimum=32,
                                maximum=512,
                                value=256,
                                step=16
                            )
                            code_temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.2,
                                step=0.1
                            )
                        
                        code_btn = gr.Button("Generate Code", variant="primary")
                    
                    # Examples Panel
                    with gr.Column(scale=1):
                        gr.Markdown("### Example Prompts")
                        code_examples = gr.Dataset(
                            components=[code_prompt],
                            samples=[[ex["prompt"]] for ex in CODE_GENERATION_EXAMPLES],
                            headers=["Prompt"],
                            label="Click an example to load it"
                        )
                
                code_output = gr.Code(
                    label="Generated Code",
                    language="python",
                    lines=12,
                    wrap=True
                )
                
                # Connect the components
                code_btn.click(
                    code_generation_handler,
                    inputs=[code_prompt, code_max_length, code_temperature],
                    outputs=code_output
                )
            
            # Creative Writing Tab
            with gr.TabItem("Creative Writing"):
                with gr.Row():
                    with gr.Column(scale=3):
                        creative_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe what you want Gemma to write...",
                            lines=3
                        )
                        
                        with gr.Row():
                            creative_max_length = gr.Slider(
                                label="Max Length",
                                minimum=64,
                                maximum=1024,
                                value=400,
                                step=32
                            )
                            creative_temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.5,
                                maximum=1.5,
                                value=1.0,
                                step=0.1
                            )
                        
                        creative_btn = gr.Button("Generate", variant="primary")
                    
                    # Examples Panel
                    with gr.Column(scale=1):
                        gr.Markdown("### Example Prompts")
                        creative_examples = gr.Dataset(
                            components=[creative_prompt],
                            samples=[[ex["prompt"]] for ex in CREATIVE_WRITING_EXAMPLES],
                            headers=["Prompt"],
                            label="Click an example to load it"
                        )
                
                creative_output = gr.Textbox(
                    label="Generated Content",
                    placeholder="Generated content will appear here...",
                    lines=10
                )
                
                # Connect the components
                creative_btn.click(
                    creative_writing_handler,
                    inputs=[creative_prompt, creative_max_length, creative_temperature],
                    outputs=creative_output
                )
        
        # Connect the model loading components
        load_button.click(
            load_model_handler,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        # Add footer with information about Gemma
        gr.Markdown(
            """
            ### About Gemma
            Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.
            
            **Resources:**
            - [Gemma on Hugging Face](https://huggingface.co/google/gemma)
            - [Google AI Gemma Overview](https://ai.google.dev/gemma)
            - [Gemma on GitHub](https://github.com/google/gemma.cpp)
            """
        )
    
    return demo

def launch_app(server_name="0.0.0.0", server_port=7860, return_app=False):
    """Launch the Gradio app with specified server settings
    
    Args:
        server_name: The hostname to listen on
        server_port: The port to listen on
        return_app: If True, return the Gradio app instead of launching it
    
    Returns:
        If return_app is True, returns the Gradio app object
        Otherwise, returns None
    """
    if not GRADIO_AVAILABLE:
        logger.error("Cannot launch app - Gradio is not available")
        print("ERROR: Gradio is not available. Please install it with: pip install gradio")
        return None
        
    logger.info(f"Initializing Gradio app for {server_name}:{server_port}")
    demo = create_gemma_demo()
    
    if return_app:
        return demo
    
    logger.info(f"Starting Gradio app on {server_name}:{server_port}")
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        favicon_path="./static/img/gemma-logo.svg"
    )

if __name__ == "__main__":
    launch_app()