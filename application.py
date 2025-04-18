import os
import sys

# Ensure the current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import from your actual app file
from app import app, server as application

# This is important: Azure will look for the 'application' variable
if __name__ == '__main__':
    application.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))