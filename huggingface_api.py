"""
Hugging Face API client for the Gemma Demo application.

This module provides a simple API client for the Hugging Face Inference API,
using only the standard library to avoid dependency issues.
"""

import os
import json
import logging
import urllib.request
import urllib.error
import urllib.parse

# Configure logging
logger = logging.getLogger(__name__)

class HuggingFaceAPIClient:
    """
    A simple client for the Hugging Face Inference API using only standard library.
    """
    def __init__(self):
        self.token = os.environ.get("HUGGING_FACE_TOKEN")
        self.api_url = "https://api-inference.huggingface.co/models/google/"
        self.model_name = None
        
        if not self.token:
            logger.warning("No Hugging Face token found in environment")
        else:
            logger.info("Hugging Face token found")
    
    def load_model(self, model_name):
        """Set the model to use for generation"""
        self.model_name = model_name
        logger.info(f"Set model to: {model_name}")
        return {"status": "success", "message": f"Model {model_name} ready for use"}
    
    def generate_text(self, prompt, max_length=256, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1):
        """Generate text using the Hugging Face Inference API"""
        if not self.token:
            return {"error": "No Hugging Face token found. Please set the HUGGING_FACE_TOKEN environment variable."}
        
        if not self.model_name:
            return {"error": "No model loaded. Please load a model first."}
        
        # Build the API URL
        url = f"{self.api_url}{self.model_name}"
        
        # Prepare the payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty
            }
        }
        
        try:
            # Convert payload to JSON
            data = json.dumps(payload).encode('utf-8')
            
            # Create request
            req = urllib.request.Request(url, data=data, method='POST')
            req.add_header('Authorization', f'Bearer {self.token}')
            req.add_header('Content-Type', 'application/json')
            
            # Send request
            logger.info(f"Sending request to {url}")
            with urllib.request.urlopen(req) as response:
                response_data = response.read().decode('utf-8')
                result = json.loads(response_data)
                
                # Extract generated text
                if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                    generated_text = result[0]['generated_text']
                    return {"text": generated_text, "source": "actual_model"}
                else:
                    logger.error(f"Unexpected response format: {result}")
                    return {"error": "Unexpected response format"}
        
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            logger.error(f"HTTP error: {e.code} - {error_body}")
            return {"error": f"HTTP error {e.code}: {error_body}"}
        
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return {"error": str(e)}
    
    def answer_question(self, question, max_length=150, temperature=0.4):
        """Answer a question using the model"""
        # Add instructions to make the model answer a question
        prompt = f"Please answer the following question:\n{question}"
        return self.generate_text(prompt, max_length, temperature)
    
    def generate_code(self, prompt, max_length=256, temperature=0.2):
        """Generate code based on the given prompt"""
        # Add instructions to generate code
        code_prompt = f"Generate code for: {prompt}\n```"
        return self.generate_text(code_prompt, max_length, temperature)
    
    def creative_writing(self, prompt, max_length=400, temperature=1.0):
        """Generate creative writing based on the prompt"""
        # Add instructions for creative writing
        creative_prompt = f"Create a creative piece based on the following: {prompt}"
        return self.generate_text(creative_prompt, max_length, temperature)

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = HuggingFaceAPIClient()
    
    # Test if token is available
    if client.token:
        print("Token found, testing API...")
        
        # Load model
        client.load_model("gemma-7b")
        
        # Generate text
        result = client.generate_text("Hello, I am")
        print(f"API Response: {result}")
    else:
        print("No token found. Please set the HUGGING_FACE_TOKEN environment variable.")