"""
Main module for the Gemma Demo application.

This is the entry point for the Gemma demo application.
We use a dedicated API-optimized implementation for direct Hugging Face API integration.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import the optimized API-only Flask implementation
try:
    from api_only_app import app
    logger.info("API-optimized implementation imported successfully")
    has_api_app = True
except ImportError as e:
    logger.error(f"Could not import API-optimized app: {str(e)}")
    has_api_app = False

# Try to import the regular Flask app as a fallback
if not has_api_app:
    try:
        from app import app as flask_app
        has_flask_fallback = True
        logger.info("Regular Flask app imported as fallback")
        app = flask_app
    except ImportError:
        has_flask_fallback = False
        logger.error("Could not import Flask app!")
        sys.exit(1)

if __name__ == "__main__":
    # For direct execution (not through gunicorn)
    port = int(os.environ.get("PORT", 5000))
    
    if has_api_app:
        from api_only_app import launch_app
        logger.info(f"Starting API-optimized app on port {port}...")
        launch_app(server_name="0.0.0.0", server_port=port)
    elif has_flask_fallback:
        logger.info(f"Starting Flask app on port {port}...")
        flask_app.run(host="0.0.0.0", port=port, debug=True)
    else:
        logger.error("No application could be imported!")
        sys.exit(1)