# Minimal Agentic LLM Builder - Python Desktop MVP

A lightweight Python desktop application for loading local small LLMs, sending prompts, inspecting responses, and running evaluations.

## Features

- **Model Loading**: Load GGUF format models via llama-cpp-python
- **Interactive Prompting**: System and user prompt interface with response viewer
- **Evaluation Runner**: Batch evaluation with pass/fail metrics and CSV export
- **Configuration Management**: Persistent settings via YAML
- **Minimal Dependencies**: PySide6, PyYAML, llama-cpp-python only

## Project Structure

```
├── main.py              # Main application with PySide6 GUI
├── requirements.txt     # Python dependencies
├── build.py            # PyInstaller build script
├── config.yaml         # Application configuration
├── eval.yaml           # Sample evaluation test cases
├── prompts/            # Example prompt files
│   ├── example1.md     # Basic assistant prompt
│   └── example2.md     # Code generation prompt
└── README.md           # This file
```

## Installation

1. **Create virtual environment:**
   ```bash
   python -m venv llm-builder-env
   source llm-builder-env/bin/activate  # On Windows: llm-builder-env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download a GGUF model:**
   - Visit Hugging Face and search for "gguf" models
   - Download a small model like `Llama-2-7B-Chat-GGUF` or similar
   - Place the .gguf file in an accessible location

## Usage

### Running the Application

```bash
python main.py
```

### Basic Workflow

1. **Load Model:**
   - Go to the "Model" tab
   - Browse and select your GGUF model file
   - Adjust settings (threads, context length, temperature)
   - Click "Load Model"

2. **Chat with Model:**
   - Switch to "Prompt & Run" tab
   - Enter system prompt (optional) and user prompt
   - Click "Run Prompt" to generate response
   - View response and generation statistics

3. **Run Evaluations:**
   - Switch to "Evaluation" tab
   - Load or create an eval.yaml file with test cases
   - Click "Run Evaluation" to batch process
   - Export results to CSV for analysis

### Configuration

The app automatically saves your settings to `config.yaml`:
- Last loaded model path
- Model parameters (threads, context, temperature)
- UI preferences

### Evaluation Format

Create evaluation files in YAML format:

```yaml
- input: "What is 2 + 2?"
  expected: "4"

- input: "Explain Python in one sentence."
  expected: "programming language"
```

## Building Executable

To create a standalone executable:

```bash
python build.py
```

This will create a single-file executable in the `dist/` folder using PyInstaller.

## Model Requirements

- **Format**: GGUF (recommended for llama.cpp compatibility)
- **Size**: 1GB-8GB for reasonable performance on desktop
- **Sources**: 
  - Hugging Face Hub (search "gguf")
  - TheBloke's model conversions
  - Official model repositories

## System Requirements

- **Python**: 3.8+
- **RAM**: 8GB+ recommended (depends on model size)
- **Storage**: 2GB+ free space for models
- **OS**: Windows, macOS, or Linux

## Dependencies

- **PySide6**: Modern Qt-based GUI framework
- **PyYAML**: YAML configuration file handling
- **llama-cpp-python**: Local LLM inference engine
- **PyInstaller**: Executable packaging (build only)

## Troubleshooting

### Model Loading Issues
- Ensure your model file is in GGUF format
- Check that you have enough RAM for the model
- Try reducing `n_ctx` or using a smaller model

### Performance Issues
- Increase `n_threads` to match your CPU cores
- Use GPU acceleration if available (requires special llama-cpp-python build)
- Close other applications to free up RAM

### UI Issues
- Update PySide6: `pip install --upgrade PySide6`
- Check that your display scaling is not causing layout issues

## License

This project is provided as-is for educational and development purposes.

## Contributing

This is a minimal MVP implementation. Consider these areas for enhancement:
- GPU acceleration support
- Streaming response display
- Advanced evaluation metrics
- Model comparison features
- Conversation history
- Plugin system for custom evaluators
