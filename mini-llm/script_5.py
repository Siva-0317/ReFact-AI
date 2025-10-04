# Create config.yaml - Default configuration file
config_yaml_content = '''# Minimal Agentic LLM Builder Configuration
# This file stores your application settings

# Model settings
last_model_path: ""
n_threads: 4
n_ctx: 2048
temperature: 0.7
max_tokens: 512

# UI settings
window_width: 1200
window_height: 800

# Evaluation settings
default_eval_file: "eval.yaml"
'''

with open("config.yaml", "w") as f:
    f.write(config_yaml_content)

# Create eval.yaml - Sample evaluation test cases
eval_yaml_content = '''# Sample Evaluation Test Cases
# Each item should have 'input' and optionally 'expected' fields

- input: "What is 2 + 2?"
  expected: "4"

- input: "Explain what Python is in one sentence."
  expected: "programming language"

- input: "What is the capital of France?"
  expected: "Paris"

- input: "Write a simple greeting."
  expected: "hello"

- input: "What color is the sky?"
  expected: "blue"
'''

with open("eval.yaml", "w") as f:
    f.write(eval_yaml_content)

print("Created config.yaml and eval.yaml")