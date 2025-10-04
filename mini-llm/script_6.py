# Create prompts directory and example files
import os

os.makedirs("prompts", exist_ok=True)

# Create example1.md - System prompt example
example1_content = '''# System Prompt Example

You are a helpful AI assistant. You provide accurate, concise, and helpful responses to user questions. Always be polite and professional in your responses.

# User Prompt

Please explain the concept of machine learning in simple terms.
'''

with open("prompts/example1.md", "w") as f:
    f.write(example1_content)

# Create example2.md - Code generation example  
example2_content = '''# System Prompt Example

You are a Python programming expert. When asked to write code, provide clean, well-commented Python code with explanations.

# User Prompt

Write a Python function that calculates the factorial of a number using recursion.
'''

with open("prompts/example2.md", "w") as f:
    f.write(example2_content)

print("Created prompts directory with example1.md and example2.md")