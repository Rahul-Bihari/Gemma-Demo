"""Check for availability of required packages for the Gemma Demo."""
import importlib.util
import sys

def check_package(package_name):
    """Check if a package is installed."""
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None

def print_package_status(package_name):
    """Print the status of a package."""
    status = check_package(package_name)
    print(f"{package_name}: {'✓ Available' if status else '✗ Not available'}")
    return status

if __name__ == "__main__":
    print("\n=== Checking available packages ===\n")
    
    # Check all required packages
    gradio_available = print_package_status("gradio")
    flask_available = print_package_status("flask")
    streamlit_available = print_package_status("streamlit")
    torch_available = print_package_status("torch")
    transformers_available = print_package_status("transformers")
    
    print("\n=== Summary ===\n")
    
    # Determine which implementations are available
    if gradio_available and torch_available and transformers_available:
        print("✓ Gradio implementation available")
        preferred_impl = "gradio"
    else:
        print("✗ Gradio implementation unavailable (missing dependencies)")
        preferred_impl = None
    
    if streamlit_available and torch_available and transformers_available:
        print("✓ Streamlit implementation available")
        if preferred_impl is None:
            preferred_impl = "streamlit"
    else:
        print("✗ Streamlit implementation unavailable (missing dependencies)")
    
    if flask_available:
        print("✓ Flask implementation available")
        if preferred_impl is None:
            preferred_impl = "flask"
    else:
        print("✗ Flask implementation unavailable (missing dependencies)")
    
    # Print the recommended implementation
    print("\n=== Recommendation ===\n")
    if preferred_impl == "gradio":
        print("✓ Use the Gradio implementation (app_gradio.py)")
    elif preferred_impl == "streamlit":
        print("✓ Use the Streamlit implementation (app_streamlit.py)")
    elif preferred_impl == "flask":
        print("✓ Use the Flask implementation (app.py)")
    else:
        print("✗ No suitable implementation available. Please install required dependencies.")