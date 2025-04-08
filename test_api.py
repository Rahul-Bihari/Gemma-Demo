"""
Test script for the Hugging Face API client.
"""

import logging
import os
from huggingface_api import HuggingFaceAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_api_client():
    """Test the Hugging Face API client."""
    print("Testing Hugging Face API client...")
    
    # Initialize the client
    client = HuggingFaceAPIClient()
    
    # Check if token is available
    if client.token:
        print(f"Token found: {client.token[:5]}...{client.token[-5:]}")
        
        # Load model
        print("\nLoading model...")
        result = client.load_model("gemma-7b")
        print(f"Load model result: {result}")
        
        # Generate text
        print("\nGenerating text...")
        result = client.generate_text("Hello, I am")
        print(f"Generate text result: {result}")
        
        # Generate code
        print("\nGenerating code...")
        result = client.generate_code("Write a Python function to reverse a string")
        print(f"Generate code result: {result}")
        
        # Answer question
        print("\nAnswering question...")
        result = client.answer_question("What is the capital of France?")
        print(f"Answer question result: {result}")
        
        # Creative writing
        print("\nCreative writing...")
        result = client.creative_writing("Once upon a time")
        print(f"Creative writing result: {result}")
    else:
        print("No token found. Please set the HUGGING_FACE_TOKEN environment variable.")

if __name__ == "__main__":
    test_api_client()