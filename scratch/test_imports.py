import sys
import os

# Add src to path
sys.path.append(os.path.abspath('.'))

modules_to_test = [
    'src.config',
    'src.services.database',
    'src.services.shopify',
    'src.services.clickup',
    'src.agents.prompts',
    'src.agents.tools.order_lookup',
    'src.agents.tools.support_ticket',
    'src.agents.tools.knowledge_base',
    'src.agents.elena',
]

for mod in modules_to_test:
    try:
        __import__(mod)
        print(f"✅ Imported {mod}")
    except Exception as e:
        print(f"❌ Failed to import {mod}: {e}")

print("Import test finished.")
