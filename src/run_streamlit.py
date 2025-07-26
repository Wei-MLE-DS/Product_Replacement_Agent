#!/usr/bin/env python3
"""
Launcher script for the Image Classification Agent Streamlit app
"""

import os
import sys
import subprocess

def main():
    # Change to the image_detection directory
    os.chdir('image_detection')
    
    # Set environment variable to avoid OpenMP conflicts
    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'streamlit_image_agent.py',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 