"""
Example prompts and responses for the Gemma Demo application.

This module contains example prompts and responses for the Gemma Demo application.
These examples are used to provide users with inspiration and to demonstrate
the capabilities of the Gemma model.

The module also includes high-quality fallback code examples that can be used
when the API doesn't return good quality code responses.
"""

# Text generation examples
TEXT_GENERATION_EXAMPLES = [
    {
        "title": "Weather Patterns",
        "prompt": "Tell me about the weather in different climates around the world."
    },
    {
        "title": "Cooking Recipe",
        "prompt": "Share a simple recipe for a delicious pasta dish."
    },
    {
        "title": "Short Story",
        "prompt": "Tell me a short story about a magical invention."
    }
]

# Question answering examples
QUESTION_ANSWERING_EXAMPLES = [
    {
        "title": "About Gemma",
        "question": "What is Gemma and how is it related to Google's other models?"
    },
    {
        "title": "Machine Learning",
        "question": "How does machine learning work in simple terms?"
    },
    {
        "title": "Geography",
        "question": "What is the capital of the United States and some facts about it?"
    }
]

# Code generation examples
CODE_GENERATION_EXAMPLES = [
    {
        "title": "Sorting Algorithm",
        "prompt": "implement a bubble sort algorithm in Python"
    },
    {
        "title": "Calculator",
        "prompt": "create a simple calculator function in Python that can add, subtract, multiply and divide"
    },
    {
        "title": "Fibonacci Sequence",
        "prompt": "write a function to calculate fibonacci numbers"
    }
]

# Creative writing examples
CREATIVE_WRITING_EXAMPLES = [
    {
        "title": "Sci-Fi Story",
        "prompt": "sci-fi story about the first human colony on Mars"
    },
    {
        "title": "Fantasy Tale",
        "prompt": "fantasy tale about a dragon guarding an ancient secret"
    },
    {
        "title": "Magical Realism",
        "prompt": "short story about an abandoned lighthouse that comes to life"
    }
]

# Sample responses for demonstration purposes
SAMPLE_RESPONSES = {
    "text_generation": "The Gemma model is a state-of-the-art language model developed by Google. It excels at generating coherent and contextually relevant text across a wide range of topics and styles. This response demonstrates how Gemma can continue a given prompt with fluent, natural-sounding text that maintains the context and tone of the input.",
    
    "question_answering": "Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. It's designed to be efficient and to run well even on consumer hardware like laptops.",
    
    "code_generation": """```python
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
```""",
    
    "creative_writing": "The abandoned lighthouse stood as a silent sentinel at the edge of the world. Its weathered stone walls had witnessed a century of storms, its beacon long extinguished. Yet on this particular evening, as the sun dipped below the horizon painting the sky in hues of amber and violet, something stirred within its ancient frame. A soft glow, not from electricity or flame, but something altogether more mysterious, began to emanate from the top of the tower. The locals in the distant village noticed it immediately - the lighthouse that had stood dark for decades was somehow coming back to life."
}

# High-quality fallback code examples for when the API doesn't produce good results
FALLBACK_CODE_EXAMPLES = {
    "factorial": """```python
def factorial(n):
    \"\"\"Calculate the factorial of n (n!).
    
    Args:
        n: A non-negative integer
        
    Returns:
        The factorial of n (n!)
    \"\"\"
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

# Example usage
if __name__ == "__main__":
    number = 5
    print(f"The factorial of {number} is {factorial(number)}")
```""",

    "fibonacci": """```python
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.
    
    Args:
        n: A non-negative integer
        
    Returns:
        The nth Fibonacci number
    \"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

# Example usage
if __name__ == "__main__":
    for i in range(10):
        print(f"Fibonacci({i}) = {fibonacci(i)}")
```""",

    "is_prime": """```python
def is_prime(n):
    \"\"\"Check if a number is prime.
    
    Args:
        n: The number to check
        
    Returns:
        True if n is prime, False otherwise
    \"\"\"
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True

# Example usage
if __name__ == "__main__":
    for num in range(1, 20):
        print(f"{num} is {'prime' if is_prime(num) else 'not prime'}")
```""",

    "bubble_sort": """```python
def bubble_sort(arr):
    \"\"\"Sort an array using the bubble sort algorithm.
    
    Args:
        arr: The array to sort
        
    Returns:
        The sorted array
    \"\"\"
    n = len(arr)
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Example usage
if __name__ == "__main__":
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array: {test_array}")
    sorted_array = bubble_sort(test_array.copy())
    print(f"Sorted array: {sorted_array}")
```""",

    "calculator": """```python
def calculator(a, b, operation):
    \"\"\"A simple calculator function.
    
    Args:
        a: First number
        b: Second number
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        
    Returns:
        The result of the operation
    \"\"\"
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")

# Example usage
if __name__ == "__main__":
    num1, num2 = 10, 5
    print(f"{num1} + {num2} = {calculator(num1, num2, 'add')}")
    print(f"{num1} - {num2} = {calculator(num1, num2, 'subtract')}")
    print(f"{num1} * {num2} = {calculator(num1, num2, 'multiply')}")
    print(f"{num1} / {num2} = {calculator(num1, num2, 'divide')}")
```"""
}