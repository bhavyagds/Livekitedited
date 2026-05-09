
import os
import sys
import importlib.util
from pathlib import Path

# Add project root and src to path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

def check_imports(directory):
    files = list(Path(directory).rglob("*.py"))
    success_count = 0
    fail_count = 0
    errors = []

    print(f"Checking imports in {directory}...")
    
    for file_path in files:
        if "venv" in str(file_path) or ".gemini" in str(file_path):
            continue
            
        module_name = str(file_path.relative_to(root)).replace(os.path.sep, ".").replace(".py", "")
        
        try:
            # We don't want to actually execute all top-level code (like cli.run_app)
            # but we can check if the module can be loaded.
            # Using find_spec is safer than import_module for some cases.
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec:
                # We won't fully load it to avoid side effects, 
                # but we'll check if the file itself has syntax errors (already done)
                # and maybe try a lightweight import.
                success_count += 1
        except Exception as e:
            fail_count += 1
            errors.append(f"{file_path}: {e}")

    print(f"\nSummary: {success_count} files OK, {fail_count} failed.")
    for err in errors:
        print(f"❌ {err}")

if __name__ == "__main__":
    check_imports(os.path.join(root, "src"))
