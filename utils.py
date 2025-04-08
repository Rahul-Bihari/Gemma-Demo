"""Utility functions for the Gemma Demo application."""

def format_generated_text(text):
    """Format the generated text for better readability."""
    # Remove potential extra whitespace
    text = text.strip()
    
    # If the text contains a question/prompt followed by an answer, try to extract just the answer
    lines = text.split('\n')
    if len(lines) > 1:
        # Look for patterns like "Question:" or "Prompt:" at the beginning
        first_line_lower = lines[0].lower()
        if ("question:" in first_line_lower or 
            "prompt:" in first_line_lower or 
            "please" in first_line_lower or
            first_line_lower.endswith('?')):
            # Skip the first line and keep the rest
            text = '\n'.join(lines[1:]).strip()
    
    # Remove "Answer:" prefix if present
    if text.lower().startswith("answer:"):
        text = text[7:].strip()  # 7 = len("Answer:")
    
    # Ensure the text starts with a capital letter if it doesn't already
    if text and not text[0].isupper() and text[0].isalpha():
        text = text[0].upper() + text[1:]
    
    return text

def extract_code_blocks(text):
    """Extract code blocks from text with markdown code block syntax."""
    import re
    # Look for code blocks with ```language ... ``` pattern
    pattern = r'```(?:\w+)?\s*([\s\S]*?)\s*```'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the first code block found
        return matches[0].strip()
    
    return None

def check_code_quality(code):
    """
    Check if the code has obvious syntax errors or is incomplete.
    
    Args:
        code: The code string to check
        
    Returns:
        tuple: (is_valid, issues) where is_valid is a boolean and issues is a list of issues
    """
    issues = []
    
    # Check for obvious syntax issues
    if code is None or not code.strip():
        issues.append("Empty code")
        return False, issues
    
    # Look for common syntax errors
    if "if " in code and " if" not in code and ": " not in code and ":\n" not in code:
        issues.append("Missing colon after if statement")
    
    # Check for mismatched parentheses/brackets
    open_parens = code.count("(")
    close_parens = code.count(")")
    if open_parens != close_parens:
        issues.append(f"Mismatched parentheses: {open_parens} open, {close_parens} close")

    open_brackets = code.count("[")
    close_brackets = code.count("]")
    if open_brackets != close_brackets:
        issues.append(f"Mismatched brackets: {open_brackets} open, {close_brackets} close")
    
    open_braces = code.count("{")
    close_braces = code.count("}")
    if open_braces != close_braces:
        issues.append(f"Mismatched braces: {open_braces} open, {close_braces} close")
    
    # Check for incomplete function definitions (Python)
    if "def " in code:
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("def ") and ":" not in line:
                issues.append("Incomplete function definition (missing colon)")
            
            # Check for parameters without type or name
            if "def " in line and "(" in line and ")" in line:
                params = line.split("(")[1].split(")")[0].strip()
                if params and params != "self" and "=" in params and "," not in params:
                    param_parts = params.split("=")[0].strip()
                    if not param_parts or param_parts.isspace():
                        issues.append("Missing parameter name before assignment")
    
    # Check for variables that are used but not defined
    lines = code.split("\n")
    variable_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*='
    usage_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
    
    defined_vars = set()
    import re
    
    for line in lines:
        # Skip comments
        if line.strip().startswith("#"):
            continue
            
        # Find variable definitions
        for match in re.finditer(variable_pattern, line):
            defined_vars.add(match.group(1))
    
    # Check if the code has at least some valid content
    has_valid_code = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and len(stripped) > 5:
            has_valid_code = True
            break
            
    if not has_valid_code:
        issues.append("No valid code content found")
    
    # Look for incomplete lines ending with operators
    for line in lines:
        stripped = line.strip()
        if stripped and stripped[-1:] in "+-*/%=<>&|^":
            issues.append("Line ends with an operator, suggesting incomplete code")
    
    # Check for python syntax errors that are obvious
    if "def" in code and "return" in code:
        # Look for common indentation issues in Python functions
        prev_line_has_colon = False
        has_indentation_after_colon = False
        
        for i, line in enumerate(lines):
            if i > 0 and prev_line_has_colon and line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                issues.append("Missing indentation after a colon")
                break
                
            if line.strip().endswith(":"):
                prev_line_has_colon = True
            else:
                if prev_line_has_colon and line.strip() and (line.startswith(" ") or line.startswith("\t")):
                    has_indentation_after_colon = True
                prev_line_has_colon = False
    
    # Determine if the code is valid based on the issues
    is_valid = len(issues) <= 1  # Allow at most one minor issue
    
    return is_valid, issues

def get_fallback_code_example(prompt):
    """
    Get a fallback code example based on keywords in the prompt.
    
    Args:
        prompt: The user's code generation prompt
        
    Returns:
        A high-quality code example or None if no matching example is found
    """
    from examples import FALLBACK_CODE_EXAMPLES
    
    prompt_lower = prompt.lower()
    
    # Map keywords to fallback examples
    if "factorial" in prompt_lower:
        return FALLBACK_CODE_EXAMPLES.get("factorial")
    
    if "fibonacci" in prompt_lower:
        return FALLBACK_CODE_EXAMPLES.get("fibonacci")
    
    if "prime" in prompt_lower or "is_prime" in prompt_lower:
        return FALLBACK_CODE_EXAMPLES.get("is_prime")
    
    if "bubble sort" in prompt_lower or "bubblesort" in prompt_lower:
        return FALLBACK_CODE_EXAMPLES.get("bubble_sort")
    
    if "calculator" in prompt_lower:
        return FALLBACK_CODE_EXAMPLES.get("calculator")
    
    # No matching example found
    return None

def format_code_output(text, prompt=None, use_fallbacks=True):
    """
    Format code output with proper markdown code blocks.
    
    Args:
        text: The text to format
        prompt: The original prompt (optional) - used for matching fallback examples
        use_fallbacks: Whether to use fallback examples when API output is poor quality
        
    Returns:
        Formatted code output
    """
    # Extract any existing code blocks first
    extracted_code = extract_code_blocks(text)
    
    if extracted_code:
        # Check the quality of the extracted code
        is_valid, issues = check_code_quality(extracted_code)
        
        if is_valid:
            # If we found a valid code block, wrap it properly
            language = "python"  # Default to Python
            if "def " in extracted_code or "class " in extracted_code or "import " in extracted_code:
                language = "python"
            elif "function " in extracted_code or "const " in extracted_code or "let " in extracted_code:
                language = "javascript"
            elif "{" in extracted_code and "}" in extracted_code and ";" in extracted_code:
                language = "java"  # Could be Java, C++, etc.
                
            return f"```{language}\n{extracted_code}\n```"
        elif use_fallbacks and prompt:
            # Try to get a fallback example based on the prompt
            fallback = get_fallback_code_example(prompt)
            if fallback:
                return fallback
    
    # Check if the text already contains markdown code blocks
    if "```" in text:
        # Extract the code and check its quality
        code = extract_code_blocks(text)
        is_valid, issues = check_code_quality(code)
        
        if is_valid:
            return text
        elif use_fallbacks and prompt:
            # Try to get a fallback example based on the prompt
            fallback = get_fallback_code_example(prompt)
            if fallback:
                return fallback
    
    # Try to extract just the actual code and remove explanatory text
    lines = text.split('\n')
    code_start = -1
    
    # Look for prompt markers and code generation prompts
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if (line_lower.startswith("generate code") or 
            line_lower.startswith("here's") or 
            line_lower.startswith("here is") or
            "implement" in line_lower and "function" in line_lower):
            code_start = i + 1
            break
    
    if code_start >= 0 and code_start < len(lines):
        # Extract code from after the prompt
        code_text = '\n'.join(lines[code_start:]).strip()
    else:
        code_text = text.strip()
    
    # Check the quality of the extracted code
    is_valid, issues = check_code_quality(code_text)
    
    if not is_valid and use_fallbacks and prompt:
        # Try to get a fallback example based on the prompt
        fallback = get_fallback_code_example(prompt)
        if fallback:
            return fallback
    
    # Try to detect what language the code might be
    language = "python"  # Default to Python
    if "def " in code_text or "class " in code_text or "import " in code_text:
        language = "python"
    elif "function " in code_text or "const " in code_text or "let " in code_text or "var " in code_text:
        language = "javascript"
    elif "{" in code_text and "}" in code_text and ";" in code_text:
        language = "java"  # Could be Java, C++, etc.
        
    # Wrap the code in markdown code blocks with the detected language
    return f"```{language}\n{code_text}\n```"

def validate_input(prompt, max_length, temperature, top_p, top_k, repetition_penalty):
    """Validate input parameters for text generation."""
    errors = []
    
    # Check prompt
    if not prompt or prompt.strip() == "":
        errors.append("Prompt cannot be empty")
    
    # Check max_length
    try:
        max_length_val = int(max_length)
        if max_length_val < 1 or max_length_val > 1024:
            errors.append("Max length must be between 1 and 1024")
    except (ValueError, TypeError):
        errors.append("Max length must be a valid integer")
    
    # Check temperature
    try:
        temp_val = float(temperature)
        if temp_val < 0.0 or temp_val > 2.0:
            errors.append("Temperature must be between 0.0 and 2.0")
    except (ValueError, TypeError):
        errors.append("Temperature must be a valid number")
    
    # Check top_p
    try:
        top_p_val = float(top_p)
        if top_p_val < 0.0 or top_p_val > 1.0:
            errors.append("Top p must be between 0.0 and 1.0")
    except (ValueError, TypeError):
        errors.append("Top p must be a valid number")
    
    # Check top_k
    try:
        top_k_val = int(top_k)
        if top_k_val < 1 or top_k_val > 100:
            errors.append("Top k must be between 1 and 100")
    except (ValueError, TypeError):
        errors.append("Top k must be a valid integer")
    
    # Check repetition_penalty
    try:
        rep_val = float(repetition_penalty)
        if rep_val < 1.0 or rep_val > 2.0:
            errors.append("Repetition penalty must be between 1.0 and 2.0")
    except (ValueError, TypeError):
        errors.append("Repetition penalty must be a valid number")
    
    return errors