# Let me create the basic project structure and core files for the LLM Builder app
import os
import json

# Create project structure
project_structure = {
    "main.py": "Main application file with PySide6 GUI",
    "requirements.txt": "Python dependencies",
    "build.py": "PyInstaller build script", 
    "config.yaml": "Default configuration file",
    "eval.yaml": "Sample evaluation test cases",
    "prompts/": {
        "example1.md": "Example system prompt",
        "example2.md": "Example user prompt"
    }
}

def create_structure(structure, base_path=""):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            # It's a directory
            os.makedirs(path, exist_ok=True)
            create_structure(content, path)
        else:
            # It's a file - we'll create the content later
            print(f"File: {path} - {content}")

print("Project Structure:")
create_structure(project_structure)