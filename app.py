"""
Gemma Demo Application

A Flask-based demo for Gemma, Google's state-of-the-art lightweight open models.
This demo uses the Hugging Face Inference API to access the actual Gemma model,
without requiring the full transformers library to be installed.
"""

import os
import json
import logging
from flask import Flask, render_template, request, jsonify, session

# Import our examples and utility functions
from examples import (
    TEXT_GENERATION_EXAMPLES,
    QUESTION_ANSWERING_EXAMPLES,
    CODE_GENERATION_EXAMPLES,
    CREATIVE_WRITING_EXAMPLES,
    SAMPLE_RESPONSES
)
from utils import (
    format_generated_text, 
    format_code_output, 
    validate_input, 
    extract_code_blocks, 
    check_code_quality, 
    get_fallback_code_example
)

# Import the HuggingFace API client
try:
    from huggingface_api import HuggingFaceAPIClient
    HUGGINGFACE_API_AVAILABLE = True
except ImportError:
    HUGGINGFACE_API_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "gemma-demo-secret")

class GemmaDemo:
    """
    Implementation of the Gemma demo.
    Uses the Hugging Face Inference API if available, 
    otherwise falls back to a simulated implementation.
    """
    def __init__(self):
        """Initialize the GemmaDemo instance."""
        self.model_name = None
        self.loaded = False
        self.using_real_model = False
        self.hf_client = None
        
        # Check if we can use the Hugging Face API
        if HUGGINGFACE_API_AVAILABLE:
            logger.info("Hugging Face API client is available")
            self.hf_client = HuggingFaceAPIClient()
            logger.info(f"HF client token status: {'Token present' if self.hf_client.token else 'No token'}")
            
            if self.hf_client.token:
                logger.info("Hugging Face token found - will use actual Gemma model via API")
                self.using_real_model = True
            else:
                logger.warning("Hugging Face token not found - using simulated responses")
        else:
            logger.warning("HuggingFace API client not available - using simulated responses")
            
        logger.info(f"Initialized GemmaDemo, using_real_model={self.using_real_model}")

    def load_model(self, model_name):
        """Load the Gemma model via Hugging Face API or simulate loading"""
        logger.info(f"Load model called with model_name={model_name}, using_real_model={self.using_real_model}, client_exists={self.hf_client is not None}")
        
        # If we can use the real model via API, try to do so
        if self.using_real_model and self.hf_client:
            try:
                logger.info(f"Setting model via API: {model_name}")
                result = self.hf_client.load_model(model_name)
                logger.info(f"API load_model result: {result}")
                
                if "error" in result:
                    logger.error(f"Error loading model via API: {result['error']}")
                    logger.info("Falling back to simulated responses")
                    self.using_real_model = False
                    self.loaded = False  # Important: Mark as not loaded for real model
                else:
                    self.model_name = model_name
                    self.loaded = True   # Mark as successfully loaded
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
                self.loaded = False      # Important: Mark as not loaded for real model
                # Fall through to simulated response
        else:
            logger.warning(f"Skipping real model loading: using_real_model={self.using_real_model}, hf_client={self.hf_client is not None}")
        
        # Simulate loading a model
        self.model_name = model_name
        # We only want to treat the model as loaded if using_real_model is True
        # This ensures we don't use the actual model capabilities with simulated responses
        self.loaded = False if self.using_real_model else True
        logger.info(f"Simulating model loading: {model_name}, loaded status={self.loaded}")
        
        return {
            "status": "warning", 
            "message": f"Simulated model {model_name} loaded. Note: Using sample responses, not the actual Gemma model.",
            "using_real_model": False
        }

    def generate_text(self, prompt, max_length=256, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1):
        """Generate text based on parameters"""
        logger.info(f"Generating text for prompt: {prompt[:50]}...")
        logger.info(f"Generation parameters: using_real_model={self.using_real_model}, client_exists={self.hf_client is not None}, loaded={self.loaded}")
        
        # If we have the real model loaded via API, use it
        if self.using_real_model and self.hf_client and self.loaded:
            try:
                # Use the API client to generate text
                logger.info("Attempting to generate text via API")
                result = self.hf_client.generate_text(
                    prompt=prompt,
                    max_length=int(max_length),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty)
                )
                logger.info(f"API generate_text result keys: {result.keys()}")
                
                # Check for errors
                if "error" in result:
                    logger.error(f"Error from API: {result['error']}")
                    logger.info("Falling back to simulated response")
                else:
                    # Format and clean up the response
                    logger.info("Using actual model response")
                    formatted_response = format_generated_text(result["text"])
                    
                    return {
                        "status": "success",
                        "generated_text": formatted_response,
                        "model_name": self.model_name,
                        "source": "actual_model"
                    }
            except Exception as e:
                logger.error(f"Error using API for text generation: {str(e)}")
                logger.info("Falling back to simulated response")
        else:
            logger.warning(f"Not using real model for text generation. Conditions not met: using_real_model={self.using_real_model}, has_client={self.hf_client is not None}, loaded={self.loaded}")
                
        # If we're here, either the model isn't loaded or an error occurred
        # Simulate generation process
        logger.info("Using simulated response for text generation")
        response = SAMPLE_RESPONSES["text_generation"]
        formatted_response = format_generated_text(response)
        
        return {
            "status": "success",
            "generated_text": formatted_response,
            "model_name": self.model_name,
            "source": "simulated"
        }

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
                logger.error(f"Error using API for Q&A: {str(e)}")
                logger.info("Falling back to simulated response")
        
        # If we're here, either the model isn't loaded or an error occurred
        # Simulate question answering
        response = SAMPLE_RESPONSES["question_answering"]
        formatted_response = format_generated_text(response)
        
        return {
            "status": "success",
            "answer": formatted_response,
            "model_name": self.model_name,
            "source": "simulated"
        }

    def generate_code(self, prompt, max_length=256, temperature=0.2):
        """Generate code based on the given prompt"""
        logger.info(f"Generating code for prompt: {prompt[:50]}...")
        
        # If we have the real model loaded via API, use it
        if self.using_real_model and self.hf_client and self.loaded:
            try:
                # Use the API client to generate code
                logger.info(f"Attempting to generate code via API")
                result = self.hf_client.generate_code(
                    prompt=prompt,
                    max_length=int(max_length),
                    temperature=float(temperature)
                )
                logger.info(f"API generate_code result keys: {result.keys()}")
                
                # Check for errors
                if "error" in result:
                    logger.error(f"Error from API: {result['error']}")
                    logger.info("Falling back to high-quality example")
                else:
                    # Format and clean up the response
                    raw_text = result["text"]
                    
                    # Check the code quality before formatting
                    extracted_code = extract_code_blocks(raw_text)
                    if extracted_code:
                        is_valid, issues = check_code_quality(extracted_code)
                        if not is_valid:
                            logger.warning(f"API returned low-quality code with issues: {issues}")
                    
                    # Format with potential fallback to examples if quality is poor
                    formatted_code = format_code_output(raw_text, prompt=prompt, use_fallbacks=True)
                    
                    # Check if we're using a fallback
                    fallback = get_fallback_code_example(prompt)
                    if fallback and formatted_code == fallback:
                        logger.info("Using fallback code example due to poor API output quality")
                        source = "enhanced_fallback"
                    else:
                        logger.info("Using actual model response")
                        source = "actual_model"
                    
                    return {
                        "status": "success",
                        "code": formatted_code,
                        "model_name": self.model_name,
                        "source": source
                    }
            except Exception as e:
                logger.error(f"Error using API for code generation: {str(e)}")
                logger.info("Falling back to high-quality example")
        
        # Try to get a matching high-quality example based on the prompt
        fallback = get_fallback_code_example(prompt)
        if fallback:
            logger.info("Using fallback code example")
            return {
                "status": "success",
                "code": fallback,
                "model_name": self.model_name,
                "source": "high_quality_example"
            }
            
        # If no matching example, use the default sample response
        logger.info("No matching fallback example, using default sample")
        response = SAMPLE_RESPONSES["code_generation"]
        formatted_response = format_code_output(response)
        
        return {
            "status": "success",
            "code": formatted_response,
            "model_name": self.model_name,
            "source": "simulated"
        }
        
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
        # Simulate creative writing
        response = SAMPLE_RESPONSES["creative_writing"]
        formatted_response = format_generated_text(response)
        
        return {
            "status": "success",
            "text": formatted_response,
            "model_name": self.model_name,
            "source": "simulated"
        }

# Initialize the demo - create a new instance
logger.info("Initializing GemmaDemo")
gemma_demo = GemmaDemo()

# Force load the model at startup
if gemma_demo.using_real_model and gemma_demo.hf_client:
    logger.info("Auto-loading default model at startup")
    result = gemma_demo.load_model("gemma-7b")
    logger.info(f"Auto-load result: {result}")
    logger.info(f"Demo state after auto-load: loaded={gemma_demo.loaded}, using_real_model={gemma_demo.using_real_model}")

@app.route('/')
def index():
    """Render the main demo page"""
    # Make the examples available to the template
    return render_template(
        'index.html',
        text_examples=TEXT_GENERATION_EXAMPLES,
        qa_examples=QUESTION_ANSWERING_EXAMPLES,
        code_examples=CODE_GENERATION_EXAMPLES,
        creative_examples=CREATIVE_WRITING_EXAMPLES
    )

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """API endpoint to load a model"""
    logger.info("load_model API endpoint called")
    data = request.json
    model_name = data.get('model_name', 'gemma-7b')
    logger.info(f"API load_model: using model_name={model_name}")
    
    # Load the model
    result = gemma_demo.load_model(model_name)
    logger.info(f"API load_model result: {result}")
    
    # Check the loaded status after model loading
    logger.info(f"After load_model, gemma_demo.loaded={gemma_demo.loaded}, using_real_model={gemma_demo.using_real_model}")
    
    # Store the selected model in the session
    session['model_name'] = model_name
    
    return jsonify(result)

@app.route('/api/generate_text', methods=['POST'])
def generate_text():
    """API endpoint for text generation"""
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 256)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', 50)
    repetition_penalty = data.get('repetition_penalty', 1.1)
    
    # Validate inputs
    errors = validate_input(prompt, max_length, temperature, top_p, top_k, repetition_penalty)
    if errors:
        return jsonify({"status": "error", "errors": errors})
    
    # Generate text
    result = gemma_demo.generate_text(
        prompt=prompt,
        max_length=int(max_length),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty)
    )
    
    return jsonify(result)

@app.route('/api/answer_question', methods=['POST'])
def answer_question():
    """API endpoint for question answering"""
    data = request.json
    question = data.get('question', '')
    max_length = data.get('max_length', 150)
    temperature = data.get('temperature', 0.4)
    
    # Validate inputs
    if not question or question.strip() == "":
        return jsonify({"status": "error", "errors": ["Question cannot be empty"]})
    
    # Answer the question
    result = gemma_demo.answer_question(
        question=question,
        max_length=int(max_length),
        temperature=float(temperature)
    )
    
    return jsonify(result)

@app.route('/api/generate_code', methods=['POST'])
def generate_code():
    """API endpoint for code completion"""
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 256)
    temperature = data.get('temperature', 0.2)
    
    # Validate inputs
    if not prompt or prompt.strip() == "":
        return jsonify({"status": "error", "errors": ["Prompt cannot be empty"]})
    
    # Generate code
    result = gemma_demo.generate_code(
        prompt=prompt,
        max_length=int(max_length),
        temperature=float(temperature)
    )
    
    return jsonify(result)

@app.route('/api/creative_writing', methods=['POST'])
def creative_writing():
    """API endpoint for creative writing"""
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 400)
    temperature = data.get('temperature', 1.0)
    
    # Validate inputs
    if not prompt or prompt.strip() == "":
        return jsonify({"status": "error", "errors": ["Prompt cannot be empty"]})
    
    # Generate creative writing
    result = gemma_demo.creative_writing(
        prompt=prompt,
        max_length=int(max_length),
        temperature=float(temperature)
    )
    
    return jsonify(result)

def launch_app(server_name="0.0.0.0", server_port=5000):
    """Launch the Flask app with specified server settings"""
    app.run(host=server_name, port=server_port)

if __name__ == "__main__":
    launch_app()