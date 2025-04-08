"""
Gemma API Demo Application

A simplified version of the Gemma demo that focuses solely on the API integration,
without requiring Gradio or other UI libraries that might cause dependency conflicts.
This implementation connects directly to the Hugging Face API for model access.
"""

import os
import logging
import json
from flask import Flask, request, jsonify, render_template, redirect, url_for
from huggingface_api import HuggingFaceAPIClient
from utils import format_generated_text, format_code_output, validate_input
from examples import (
    TEXT_GENERATION_EXAMPLES,
    QUESTION_ANSWERING_EXAMPLES,
    CODE_GENERATION_EXAMPLES,
    CREATIVE_WRITING_EXAMPLES,
    SAMPLE_RESPONSES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

class GemmaAPIDemo:
    """
    Implementation of the Gemma demo that focuses solely on the API integration.
    Uses the Hugging Face Inference API if available, 
    otherwise falls back to a simulated implementation.
    """
    def __init__(self):
        """Initialize the GemmaAPIDemo instance."""
        self.hf_client = None
        self.model_name = None
        self.loaded = False
        self.using_real_model = False
        
        # Try to set up the HuggingFace API client
        try:
            self.hf_client = HuggingFaceAPIClient()
            logger.info("Hugging Face API client is available")
            
            if self.hf_client.token:
                logger.info(f"HF client token status: Token present")
                logger.info("Hugging Face token found - will use actual Gemma model via API")
                self.using_real_model = True
            else:
                logger.warning("No Hugging Face token found - will use simulated responses")
        except Exception as e:
            logger.error(f"Error initializing Hugging Face API client: {str(e)}")
            
        logger.info(f"Initialized GemmaAPIDemo, using_real_model={self.using_real_model}")
        
        # Auto-load a default model at startup
        logger.info("Auto-loading default model at startup")
        result = self.load_model("gemma-7b")
        logger.info(f"Auto-load result: {result}")
        logger.info(f"Demo state after auto-load: loaded={self.loaded}, using_real_model={self.using_real_model}")
        
    def load_model(self, model_name):
        """Load the Gemma model via Hugging Face API or simulate loading"""
        logger.info(f"Load model called with model_name={model_name}, using_real_model={self.using_real_model}, client_exists={self.hf_client is not None}")
        
        # If we have a real API client, try to use it
        if self.using_real_model and self.hf_client:
            try:
                logger.info(f"Setting model via API: {model_name}")
                result = self.hf_client.load_model(model_name)
                logger.info(f"API load_model result: {result}")
                
                if "error" in result:
                    logger.error(f"Error loading model via API: {result.get('error')}")
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
                logger.error(f"Error in load_model: {str(e)}")
                logger.info("Falling back to simulated responses")
                self.using_real_model = False
        
        # If we're here, we're using the simulated implementation
        self.model_name = model_name
        self.loaded = True
        return {
            "status": "success", 
            "message": f"Model {model_name} simulated. NOTE: Using simulated responses.",
            "using_real_model": False
        }
    
    def generate_text(self, prompt, max_length=256, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1):
        """Generate text based on parameters"""
        logger.info(f"Text generation requested with prompt: {prompt[:50]}...")
        
        # Validate input parameters
        errors = validate_input(prompt, max_length, temperature, top_p, top_k, repetition_penalty)
        if errors:
            return {"status": "error", "errors": errors}
        
        # If we have a real API client and it's loaded, use it
        if self.using_real_model and self.hf_client and self.loaded:
            try:
                # Call the API
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
                    logger.error(f"API error: {result.get('error')}")
                    return {"status": "error", "message": f"API error: {result.get('error')}"}
                
                # Format the generated text
                formatted_text = format_generated_text(result.get("text", ""))
                
                return {
                    "status": "success",
                    "generated_text": formatted_text,
                    "source": "actual_model"
                }
            except Exception as e:
                logger.error(f"Error generating text: {str(e)}")
                return {"status": "error", "message": f"Error: {str(e)}"}
        
        # If the model isn't loaded or we're using simulated responses
        if not self.loaded:
            return {"status": "error", "message": "No model loaded. Please load a model first."}
        
        # Use simulated response
        simulated_text = SAMPLE_RESPONSES.get("text_generation", "Simulated response text")
        
        return {
            "status": "success",
            "generated_text": simulated_text,
            "source": "simulated"
        }
    
    def answer_question(self, question, max_length=150, temperature=0.4):
        """Answer a question using the model"""
        logger.info(f"Question answering requested with question: {question[:50]}...")
        
        # Validate basic input parameters
        if not question or question.strip() == "":
            return {"status": "error", "errors": ["Question cannot be empty"]}
        
        try:
            max_length = int(max_length)
            if max_length < 1 or max_length > 1024:
                return {"status": "error", "errors": ["Max length must be between 1 and 1024"]}
        except (ValueError, TypeError):
            return {"status": "error", "errors": ["Max length must be a valid integer"]}
        
        try:
            temperature = float(temperature)
            if temperature < 0.0 or temperature > 2.0:
                return {"status": "error", "errors": ["Temperature must be between 0.0 and 2.0"]}
        except (ValueError, TypeError):
            return {"status": "error", "errors": ["Temperature must be a valid number"]}
        
        # If we have a real API client and it's loaded, use it
        if self.using_real_model and self.hf_client and self.loaded:
            try:
                # Call the API
                result = self.hf_client.answer_question(
                    question=question,
                    max_length=int(max_length),
                    temperature=float(temperature)
                )
                
                # Check for errors
                if "error" in result:
                    logger.error(f"API error: {result.get('error')}")
                    return {"status": "error", "message": f"API error: {result.get('error')}"}
                
                # Format the answer
                formatted_answer = format_generated_text(result.get("text", ""))
                
                return {
                    "status": "success",
                    "answer": formatted_answer,
                    "source": "actual_model"
                }
            except Exception as e:
                logger.error(f"Error answering question: {str(e)}")
                return {"status": "error", "message": f"Error: {str(e)}"}
        
        # If the model isn't loaded or we're using simulated responses
        if not self.loaded:
            return {"status": "error", "message": "No model loaded. Please load a model first."}
        
        # Use simulated response
        simulated_answer = SAMPLE_RESPONSES.get("question_answering", "Simulated answer to question")
        
        return {
            "status": "success",
            "answer": simulated_answer,
            "source": "simulated"
        }
    
    def generate_code(self, prompt, max_length=256, temperature=0.2):
        """Generate code based on the given prompt"""
        logger.info(f"Code generation requested with prompt: {prompt[:50]}...")
        
        # Validate basic input parameters
        if not prompt or prompt.strip() == "":
            return {"status": "error", "errors": ["Prompt cannot be empty"]}
        
        try:
            max_length = int(max_length)
            if max_length < 1 or max_length > 1024:
                return {"status": "error", "errors": ["Max length must be between 1 and 1024"]}
        except (ValueError, TypeError):
            return {"status": "error", "errors": ["Max length must be a valid integer"]}
        
        try:
            temperature = float(temperature)
            if temperature < 0.0 or temperature > 2.0:
                return {"status": "error", "errors": ["Temperature must be between 0.0 and 2.0"]}
        except (ValueError, TypeError):
            return {"status": "error", "errors": ["Temperature must be a valid number"]}
        
        # If we have a real API client and it's loaded, use it
        if self.using_real_model and self.hf_client and self.loaded:
            try:
                # Call the API
                result = self.hf_client.generate_code(
                    prompt=prompt,
                    max_length=int(max_length),
                    temperature=float(temperature)
                )
                
                # Check for errors
                if "error" in result:
                    logger.error(f"API error: {result.get('error')}")
                    return {"status": "error", "message": f"API error: {result.get('error')}"}
                
                # Format the code
                formatted_code = format_code_output(result.get("text", ""), prompt=prompt)
                
                return {
                    "status": "success",
                    "code": formatted_code,
                    "source": "actual_model"
                }
            except Exception as e:
                logger.error(f"Error generating code: {str(e)}")
                return {"status": "error", "message": f"Error: {str(e)}"}
        
        # If the model isn't loaded or we're using simulated responses
        if not self.loaded:
            return {"status": "error", "message": "No model loaded. Please load a model first."}
        
        # Use simulated response
        simulated_code = SAMPLE_RESPONSES.get("code_generation", "```python\ndef example():\n    pass\n```")
        
        return {
            "status": "success",
            "code": simulated_code,
            "source": "simulated"
        }
    
    def creative_writing(self, prompt, max_length=400, temperature=1.0):
        """Generate creative writing based on the prompt"""
        logger.info(f"Creative writing requested with prompt: {prompt[:50]}...")
        
        # Validate basic input parameters
        if not prompt or prompt.strip() == "":
            return {"status": "error", "errors": ["Prompt cannot be empty"]}
        
        try:
            max_length = int(max_length)
            if max_length < 1 or max_length > 1024:
                return {"status": "error", "errors": ["Max length must be between 1 and 1024"]}
        except (ValueError, TypeError):
            return {"status": "error", "errors": ["Max length must be a valid integer"]}
        
        try:
            temperature = float(temperature)
            if temperature < 0.0 or temperature > 2.0:
                return {"status": "error", "errors": ["Temperature must be between 0.0 and 2.0"]}
        except (ValueError, TypeError):
            return {"status": "error", "errors": ["Temperature must be a valid number"]}
        
        # If we have a real API client and it's loaded, use it
        if self.using_real_model and self.hf_client and self.loaded:
            try:
                # Call the API
                result = self.hf_client.creative_writing(
                    prompt=prompt,
                    max_length=int(max_length),
                    temperature=float(temperature)
                )
                
                # Check for errors
                if "error" in result:
                    logger.error(f"API error: {result.get('error')}")
                    return {"status": "error", "message": f"API error: {result.get('error')}"}
                
                # Format the text
                formatted_text = format_generated_text(result.get("text", ""))
                
                return {
                    "status": "success",
                    "text": formatted_text,
                    "source": "actual_model"
                }
            except Exception as e:
                logger.error(f"Error generating creative writing: {str(e)}")
                return {"status": "error", "message": f"Error: {str(e)}"}
        
        # If the model isn't loaded or we're using simulated responses
        if not self.loaded:
            return {"status": "error", "message": "No model loaded. Please load a model first."}
        
        # Use simulated response
        simulated_text = SAMPLE_RESPONSES.get("creative_writing", "Simulated creative writing text")
        
        return {
            "status": "success",
            "text": simulated_text,
            "source": "simulated"
        }

# Initialize the demo instance
demo = GemmaAPIDemo()

@app.route('/')
def index():
    """Render the main demo page"""
    return render_template(
        'index.html',
        text_examples=TEXT_GENERATION_EXAMPLES,
        qa_examples=QUESTION_ANSWERING_EXAMPLES,
        code_examples=CODE_GENERATION_EXAMPLES,
        creative_examples=CREATIVE_WRITING_EXAMPLES,
        using_real_model=demo.using_real_model,
        current_model=demo.model_name
    )

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """API endpoint to load a model"""
    data = request.get_json()
    model_name = data.get('model_name', 'gemma-7b')
    
    result = demo.load_model(model_name)
    return jsonify(result)

@app.route('/api/generate_text', methods=['POST'])
def generate_text():
    """API endpoint for text generation"""
    data = request.get_json()
    
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 256)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', 50)
    repetition_penalty = data.get('repetition_penalty', 1.1)
    
    result = demo.generate_text(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty
    )
    
    return jsonify(result)

@app.route('/api/answer_question', methods=['POST'])
def answer_question():
    """API endpoint for question answering"""
    data = request.get_json()
    
    question = data.get('question', '')
    max_length = data.get('max_length', 150)
    temperature = data.get('temperature', 0.4)
    
    result = demo.answer_question(
        question=question,
        max_length=max_length,
        temperature=temperature
    )
    
    return jsonify(result)

@app.route('/api/generate_code', methods=['POST'])
def generate_code():
    """API endpoint for code completion"""
    data = request.get_json()
    
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 256)
    temperature = data.get('temperature', 0.2)
    
    result = demo.generate_code(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature
    )
    
    return jsonify(result)

@app.route('/api/creative_writing', methods=['POST'])
def creative_writing():
    """API endpoint for creative writing"""
    data = request.get_json()
    
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 400)
    temperature = data.get('temperature', 1.0)
    
    result = demo.creative_writing(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature
    )
    
    return jsonify(result)

def launch_app(server_name="0.0.0.0", server_port=5000):
    """Launch the Flask app with specified server settings"""
    app.run(host=server_name, port=server_port, debug=True)

if __name__ == "__main__":
    launch_app()