"""
Direct test of the HuggingFace API to debug code generation.
"""

import os
import json
import logging
from huggingface_api import HuggingFaceAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_code_generation():
    """Test the code generation capabilities of the API directly."""
    # Initialize the client
    client = HuggingFaceAPIClient()
    
    if not client.token:
        print("No Hugging Face token found. Please set the HUGGING_FACE_TOKEN environment variable.")
        return
    
    print(f"Token found: {client.token[:5]}...{client.token[-5:]}")
    
    # Set the model
    client.load_model("gemma-7b")
    
    # Code generation prompts to test
    prompts = [
        "create a function to calculate factorial",
        "write a Python function to check if a number is prime",
        "implement a bubble sort algorithm",
        "create a simple calculator function in Python that can add, subtract, multiply and divide"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n\n{'='*80}\nTEST {i+1}: {prompt}\n{'='*80}")
        
        # Generate code
        result = client.generate_code(prompt=prompt, temperature=0.2)
        
        # Print raw result
        print(f"\nRAW RESPONSE:\n{'-'*40}\n{result.get('text', 'No text in response')}\n{'-'*40}")
        
        # Parse and format
        print("\nCODE EXTRACTION ATTEMPT:")
        try:
            # Try to extract code using regex
            import re
            # Look for code blocks with ```language ... ``` pattern
            pattern = r'```(?:\w+)?\s*([\s\S]*?)\s*```'
            matches = re.findall(pattern, result.get('text', ''))
            
            if matches:
                # Print the first code block found
                print(f"EXTRACTED CODE:\n{'-'*40}")
                print(matches[0].strip())
                print(f"{'-'*40}")
            else:
                # Try to find lines that look like code
                lines = result.get('text', '').split('\n')
                code_lines = []
                in_code = False
                
                for line in lines:
                    if line.strip().startswith("def ") or line.strip().startswith("class "):
                        in_code = True
                    
                    if in_code:
                        code_lines.append(line)
                
                if code_lines:
                    print(f"EXTRACTED CODE (by heuristic):\n{'-'*40}")
                    print('\n'.join(code_lines))
                    print(f"{'-'*40}")
                else:
                    print("No code blocks or recognizable code found in the response.")
        except Exception as e:
            print(f"Error extracting code: {str(e)}")

if __name__ == "__main__":
    test_code_generation()