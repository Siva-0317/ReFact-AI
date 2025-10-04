#!/usr/bin/env python3
"""
PyInstaller build script for Minimal Agentic LLM Builder.
Creates a standalone executable from the Python application.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def clean_build():
    """Clean previous build artifacts."""
    paths_to_clean = ["build", "dist", "*.spec"]
    for path in paths_to_clean:
        if "*" in path:
            # Handle glob patterns
            import glob
            for file_path in glob.glob(path):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
        else:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
                else:
                    os.remove(path)
                    print(f"Removed file: {path}")

def build_app():
    """Build the application using PyInstaller."""
    print("Building Minimal Agentic LLM Builder...")

    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--name=LLMBuilder",
        "--windowed",  # Hide console window
        "--onefile",   # Single executable
        "--add-data=config.yaml;.",  # Include config file
        "--add-data=eval.yaml;.",    # Include sample eval file
        "--add-data=prompts;prompts",  # Include prompts directory
        "--hidden-import=llama_cpp",
        "--hidden-import=yaml",
        "--collect-all=llama_cpp",
        "main.py"
    ]

    # Add icon if available
    if os.path.exists("icon.ico"):
        cmd.extend(["--icon=icon.ico"])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build successful!")
        print("Executable location: dist/LLMBuilder.exe (Windows) or dist/LLMBuilder (Linux/Mac)")
        return True
    except subprocess.CalledProcessError as e:
        print("Build failed!")
        print("Error output:")
        print(e.stderr)
        return False

def create_installer_files():
    """Create additional files for distribution."""
    # Create a simple README for the dist folder
    readme_content = """# Minimal Agentic LLM Builder

## Quick Start

1. Run the LLMBuilder executable
2. Go to the "Model" tab and load a GGUF model file
3. Switch to "Prompt & Run" tab to interact with the model
4. Use the "Evaluation" tab to run batch evaluations

## Model Files

You need to download GGUF format models. Popular sources:
- Hugging Face (search for "gguf")
- TheBloke model conversions

## Configuration

The app creates a config.yaml file to remember your settings.
You can also customize eval.yaml for your evaluation test cases.

## Troubleshooting

- If the app doesn't start, check that you have the Visual C++ Redistributable installed (Windows)
- For model loading issues, ensure your model file is in GGUF format
- Check the console output for detailed error messages
"""

    dist_path = Path("dist")
    if dist_path.exists():
        with open(dist_path / "README.txt", "w") as f:
            f.write(readme_content)
        print("Created README.txt in dist folder")

def main():
    """Main build function."""
    print("=== Minimal Agentic LLM Builder Build Script ===")

    # Check if required files exist
    if not os.path.exists("main.py"):
        print("Error: main.py not found!")
        sys.exit(1)

    # Check if PyInstaller is installed
    try:
        subprocess.run(["pyinstaller", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: PyInstaller not found! Install it with: pip install pyinstaller")
        sys.exit(1)

    # Clean previous builds
    clean_build()

    # Build the application
    success = build_app()

    if success:
        create_installer_files()
        print("\n=== Build Complete ===")
        print("Your executable is ready in the 'dist' folder!")
    else:
        print("\n=== Build Failed ===")
        print("Check the error messages above and fix any issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()
