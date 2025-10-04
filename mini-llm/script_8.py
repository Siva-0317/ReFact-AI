# Let's create a summary of all created files
import os

def list_files_recursive(directory="."):
    """List all files in the project directory."""
    files = []
    for root, dirs, file_list in os.walk(directory):
        # Skip hidden directories and common build directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'build', 'dist']]
        for file in file_list:
            if not file.startswith('.') and not file.endswith('.pyc'):
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                files.append(rel_path)
    return sorted(files)

print("📁 Project Files Created:")
print("=" * 50)

files = list_files_recursive()
for file in files:
    print(f"   {file}")

print(f"\n✅ Total files: {len(files)}")

# Count lines of code in main.py
try:
    with open("main.py", "r") as f:
        lines = len(f.readlines())
    print(f"📊 Main application: {lines} lines of Python code")
except:
    print("📊 Could not count lines in main.py")

print("\n🚀 Next steps:")
print("   1. Install dependencies: pip install -r requirements.txt")  
print("   2. Download a GGUF model from Hugging Face")
print("   3. Run the app: python main.py")
print("   4. Build executable: python build.py")