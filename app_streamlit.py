"""
Gemma Demo Application

An interactive demo for Gemma, Google's state-of-the-art lightweight open models,
using Streamlit for the interface and Hugging Face Hub for model access.
"""

import os
import logging
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

# Import example prompts and utilities
from examples import (
    TEXT_GENERATION_EXAMPLES,
    QUESTION_ANSWERING_EXAMPLES,
    CODE_GENERATION_EXAMPLES,
    CREATIVE_WRITING_EXAMPLES,
)
from utils import format_generated_text, format_code_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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

def main():
    """Main function for the Streamlit app"""
    # Configure the page
    st.set_page_config(
        page_title="Gemma Demo",
        page_icon="./static/img/gemma-logo.svg",
        layout="wide"
    )
    
    # Initialize session state for storing the demo instance
    if 'demo' not in st.session_state:
        st.session_state.demo = GemmaDemo()
    
    # Header section
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("./static/img/gemma-logo.svg", width=80)
    with col2:
        st.title("Gemma Interactive Demo")
        st.markdown("Explore the capabilities of Google's state-of-the-art lightweight open model.")
    
    # Model selection sidebar
    st.sidebar.title("Model Selection")
    model_options = ["Gemma 2B", "Gemma 7B"]
    model_mapping = {
        "Gemma 2B": "gemma-2b",
        "Gemma 7B": "gemma-7b"
    }
    model_selection = st.sidebar.selectbox(
        "Select a model",
        model_options,
        index=1  # Default to Gemma 7B
    )
    model_id = model_mapping[model_selection]
    
    # Load model button
    if st.sidebar.button("Load Model"):
        with st.spinner(f"Loading {model_selection}..."):
            result = st.session_state.demo.load_model(model_id)
            if result["status"] == "success":
                st.sidebar.success(result["message"])
            else:
                st.sidebar.error(result["message"])
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About Gemma")
    st.sidebar.markdown(
        """Gemma is a family of lightweight, state-of-the-art open models from Google, 
        built from the same research and technology used to create the Gemini models."""
    )
    st.sidebar.markdown("[Gemma on Hugging Face](https://huggingface.co/google/gemma)")
    st.sidebar.markdown("[Google AI Gemma Overview](https://ai.google.dev/gemma)")
    st.sidebar.markdown("[Gemma on GitHub](https://github.com/google/gemma.cpp)")
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Text Generation", "Question Answering", "Code Generation", "Creative Writing"])
    
    # Text Generation Tab
    with tab1:
        st.header("Text Generation")
        st.markdown("Generate text based on your prompt. Adjust the parameters to control the output.")
        
        # Two columns - one for input form, one for examples
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Text generation form
            text_prompt = st.text_area(
                "Prompt",
                placeholder="Enter your prompt here...",
                height=150
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                text_max_length = st.slider(
                    "Max Length",
                    min_value=32,
                    max_value=512,
                    value=256,
                    step=16
                )
            with col_b:
                text_temperature = st.slider(
                    "Temperature",
                    min_value=0.1,
                    max_value=1.5,
                    value=0.7,
                    step=0.1
                )
            
            # Advanced options
            with st.expander("Advanced Options"):
                col_c, col_d, col_e = st.columns(3)
                with col_c:
                    text_top_p = st.slider(
                        "Top P",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.9,
                        step=0.1
                    )
                with col_d:
                    text_top_k = st.slider(
                        "Top K",
                        min_value=1,
                        max_value=100,
                        value=50,
                        step=1
                    )
                with col_e:
                    text_repetition_penalty = st.slider(
                        "Repetition Penalty",
                        min_value=1.0,
                        max_value=2.0,
                        value=1.1,
                        step=0.1
                    )
            
            # Generate button
            if st.button("Generate Text", key="text_btn"):
                if not text_prompt.strip():
                    st.error("Prompt cannot be empty.")
                else:
                    with st.spinner("Generating text..."):
                        result = st.session_state.demo.generate_text(
                            prompt=text_prompt,
                            max_length=text_max_length,
                            temperature=text_temperature,
                            top_p=text_top_p,
                            top_k=text_top_k,
                            repetition_penalty=text_repetition_penalty
                        )
                        
                        if result["status"] == "success":
                            st.markdown("### Generated Text")
                            st.write(result["generated_text"])
                        else:
                            st.error(result["message"])
        
        with col2:
            # Examples panel
            st.markdown("### Examples")
            for example in TEXT_GENERATION_EXAMPLES:
                if st.button(example["title"], key=f"text_ex_{example['title']}"):
                    # Use a callback to set the text area
                    st.session_state.text_prompt = example["prompt"]
                    st.experimental_rerun()
    
    # Question Answering Tab
    with tab2:
        st.header("Question Answering")
        st.markdown("Ask questions and get informative answers from the Gemma model.")
        
        # Two columns - one for input form, one for examples
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # QA form
            question = st.text_area(
                "Question",
                placeholder="Ask a question...",
                height=100
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                qa_max_length = st.slider(
                    "Max Answer Length",
                    min_value=32,
                    max_value=256,
                    value=150,
                    step=8
                )
            with col_b:
                qa_temperature = st.slider(
                    "Temperature",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.4,
                    step=0.1
                )
            
            # Generate button
            if st.button("Answer Question", key="qa_btn"):
                if not question.strip():
                    st.error("Question cannot be empty.")
                else:
                    with st.spinner("Answering question..."):
                        result = st.session_state.demo.answer_question(
                            question=question,
                            max_length=qa_max_length,
                            temperature=qa_temperature
                        )
                        
                        if result["status"] == "success":
                            st.markdown("### Answer")
                            st.write(result["answer"])
                        else:
                            st.error(result["message"])
        
        with col2:
            # Examples panel
            st.markdown("### Example Questions")
            for example in QUESTION_ANSWERING_EXAMPLES:
                if st.button(example["title"], key=f"qa_ex_{example['title']}"):
                    # Use a callback to set the text area
                    st.session_state.question = example["question"]
                    st.experimental_rerun()
    
    # Code Generation Tab
    with tab3:
        st.header("Code Generation")
        st.markdown("Generate code based on your requirements.")
        
        # Two columns - one for input form, one for examples
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Code form
            code_prompt = st.text_area(
                "Prompt",
                placeholder="Describe the code you want...",
                height=100
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                code_max_length = st.slider(
                    "Max Length",
                    min_value=32,
                    max_value=512,
                    value=256,
                    step=16,
                    key="code_len"
                )
            with col_b:
                code_temperature = st.slider(
                    "Temperature",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.2,
                    step=0.1,
                    key="code_temp"
                )
            
            # Generate button
            if st.button("Generate Code", key="code_btn"):
                if not code_prompt.strip():
                    st.error("Prompt cannot be empty.")
                else:
                    with st.spinner("Generating code..."):
                        result = st.session_state.demo.generate_code(
                            prompt=code_prompt,
                            max_length=code_max_length,
                            temperature=code_temperature
                        )
                        
                        if result["status"] == "success":
                            st.markdown("### Generated Code")
                            st.code(result["code"])
                        else:
                            st.error(result["message"])
        
        with col2:
            # Examples panel
            st.markdown("### Example Prompts")
            for example in CODE_GENERATION_EXAMPLES:
                if st.button(example["title"], key=f"code_ex_{example['title']}"):
                    # Use a callback to set the text area
                    st.session_state.code_prompt = example["prompt"]
                    st.experimental_rerun()
    
    # Creative Writing Tab
    with tab4:
        st.header("Creative Writing")
        st.markdown("Generate creative stories, poems, or other written content.")
        
        # Two columns - one for input form, one for examples
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Creative writing form
            creative_prompt = st.text_area(
                "Prompt",
                placeholder="Describe what you want Gemma to write...",
                height=100,
                key="creative_prompt_input"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                creative_max_length = st.slider(
                    "Max Length",
                    min_value=64,
                    max_value=1024,
                    value=400,
                    step=32
                )
            with col_b:
                creative_temperature = st.slider(
                    "Temperature",
                    min_value=0.5,
                    max_value=1.5,
                    value=1.0,
                    step=0.1
                )
            
            # Generate button
            if st.button("Generate", key="creative_btn"):
                if not creative_prompt.strip():
                    st.error("Prompt cannot be empty.")
                else:
                    with st.spinner("Generating creative writing..."):
                        result = st.session_state.demo.creative_writing(
                            prompt=creative_prompt,
                            max_length=creative_max_length,
                            temperature=creative_temperature
                        )
                        
                        if result["status"] == "success":
                            st.markdown("### Generated Content")
                            st.write(result["text"])
                        else:
                            st.error(result["message"])
        
        with col2:
            # Examples panel
            st.markdown("### Example Prompts")
            for example in CREATIVE_WRITING_EXAMPLES:
                if st.button(example["title"], key=f"creative_ex_{example['title']}"):
                    # Use a callback to set the text area
                    st.session_state.creative_prompt_input = example["prompt"]
                    st.experimental_rerun()

if __name__ == "__main__":
    main()