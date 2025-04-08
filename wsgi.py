"""
WSGI entry point for the Gemma Demo application.

This module exposes the app for gunicorn to serve.
We try to use the Gradio implementation first, with Flask as a fallback.
"""

import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import the app for gunicorn to use - import from main.py which handles the fallback logic
from main import app

# Exposes 'app' variable for gunicorn

if __name__ == "__main__":
    # This shouldn't be called directly for wsgi.py, but just in case
    from main import has_gradio, has_flask_fallback, launch_gradio_app, flask_app
    
    port = int(os.environ.get("PORT", 5000))
    
    if has_gradio:
        logger.info("Starting Gradio app...")
        launch_gradio_app(server_name="0.0.0.0", server_port=port)
    elif has_flask_fallback:
        logger.info("Starting Flask app...")
        flask_app.run(host="0.0.0.0", port=port)
    else:
        logger.error("Neither Gradio nor Flask app could be imported!")
        sys.exit(1)