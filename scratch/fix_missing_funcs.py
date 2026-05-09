import os

funcs_to_add = """
def _is_phone_number_prompt(text: str) -> bool:
    if not text: return False
    normalized = text.lower()
    return any(k in normalized for k in ["phone", "number", "τηλέφωνο", "αριθμό"])

def _is_phone_number_collection_prompt(text: str) -> bool:
    return _is_phone_number_prompt(text)

"""

def add_funcs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "_is_phone_number_prompt" not in content:
        content = content.replace('if __name__ == "__main__":', funcs_to_add + 'if __name__ == "__main__":')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

add_funcs('n:/Gdsoftwares/Livekitedited/src/agents/elena_en.py')
add_funcs('n:/Gdsoftwares/Livekitedited/src/agents/elena_el.py')
print("Added missing functions!")
