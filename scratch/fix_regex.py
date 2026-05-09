import os

def fix_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_str = r'match = re.search(r"(?i)reference number is\s+([a-z0-9-]+)", text)'
    new_str = r'match = re.search(r"(?i)reference number is\s+([#a-zA-Z0-9-]+)", text)'
    
    content = content.replace(old_str, new_str)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

fix_file('n:/Gdsoftwares/Livekitedited/src/agents/elena_en.py')
fix_file('n:/Gdsoftwares/Livekitedited/src/agents/elena_el.py')
print("Fixed!")
