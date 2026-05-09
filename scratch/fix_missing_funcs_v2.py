import os

funcs_to_add = """
def _is_phone_number_prompt(text: str) -> bool:
    if not text: return False
    normalized = text.lower()
    return any(k in normalized for k in ["phone", "number", "τηλέφωνο", "αριθμό", "mobile", "κινητό"])

def _is_phone_number_collection_prompt(text: str) -> bool:
    return _is_phone_number_prompt(text)

"""

def add_funcs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "def _is_phone_number_prompt" not in content:
        print(f"Adding functions to {file_path}")
        # Insert after the last import or at the beginning of the file (after imports)
        # We'll just insert after the first big block of imports
        if 'import logging' in content:
            content = content.replace('import logging', funcs_to_add + 'import logging', 1)
        else:
            content = funcs_to_add + content
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        print(f"Functions already exist in {file_path}")

add_funcs('n:/Gdsoftwares/Livekitedited/src/agents/elena_en.py')
add_funcs('n:/Gdsoftwares/Livekitedited/src/agents/elena_el.py')
print("Done!")
