# Create requirements.txt file
requirements_content = '''# Minimal Agentic LLM Builder Requirements
# Core GUI framework
PySide6>=6.5.0

# YAML configuration handling
PyYAML>=6.0

# Local LLM integration
llama-cpp-python>=0.2.0

# Build tool for packaging
pyinstaller>=5.13.0
'''

with open("requirements.txt", "w") as f:
    f.write(requirements_content)

print("Created requirements.txt")