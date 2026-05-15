"""
Meallion Voice AI - Elena English Agent (Patch 1 Wrapper)
This script runs the Elena English agent with a fix for the 'No' response freeze.
It applies monkey-patches to the flow handlers to unsuppress the LLM after deterministic responses,
allowing follow-up questions to be handled by the LLM immediately.

To run Elena with this fix:
    python src/agents/patch1.py

To revert:
    Simply run the original agent: python src/agents/elena_en.py
"""

import sys
import os
import asyncio
import logging

# Ensure the project root is in the python path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import the original modules
import src.agents.elena as elena
import src.agents.elena_en as elena_en
import src.agents.flows.order_flow as order_flow
import src.agents.flows.phone_flow as phone_flow
import src.agents.flows.ticket_flow as ticket_flow

logger = logging.getLogger("patch1")

# -----------------------------------------------------------------------------
# Patched Handlers
# -----------------------------------------------------------------------------

# Original function references
_orig_order_run_order = order_flow._run_order_lookup
_orig_order_run_phone = order_flow._run_phone_lookup
_orig_phone_run_phone = phone_flow._run_phone_lookup
_orig_phone_run_order = phone_flow._run_order_lookup
_orig_ticket_run_create = ticket_flow._run_create_ticket

async def patched_run_order_lookup(ctx, order_number):
    """Wrapper that unsuppresses LLM after order lookup completes."""
    await _orig_order_run_order(ctx, order_number)
    logger.info("Patch 1: Unsuppressing LLM after order lookup")
    ctx.suppress_llm(0.0)

async def patched_run_phone_lookup(ctx, phone_number):
    """Wrapper that unsuppresses LLM after phone lookup completes."""
    # This covers both order_flow and phone_flow versions depending on which is called
    if ctx.state.support_state == "checking_order":
        await _orig_order_run_phone(ctx, phone_number)
    else:
        await _orig_phone_run_phone(ctx, phone_number)
    logger.info("Patch 1: Unsuppressing LLM after phone lookup")
    ctx.suppress_llm(0.0)

async def patched_run_create_ticket(ctx):
    """Wrapper that unsuppresses LLM after ticket creation completes."""
    await _orig_ticket_run_create(ctx)
    logger.info("Patch 1: Unsuppressing LLM after ticket creation")
    ctx.suppress_llm(0.0)

# -----------------------------------------------------------------------------
# Apply Monkey-Patches
# -----------------------------------------------------------------------------

def apply_patches():
    logger.info("Applying Patch 1 to Elena flow handlers...")
    
    # Patch order_flow.py
    order_flow._run_order_lookup = patched_run_order_lookup
    order_flow._run_phone_lookup = patched_run_phone_lookup
    
    # Patch phone_flow.py
    phone_flow._run_phone_lookup = patched_run_phone_lookup
    phone_flow._run_order_lookup = patched_run_order_lookup
    
    # Patch ticket_flow.py
    ticket_flow._run_create_ticket = patched_run_create_ticket
    
    logger.info("Patch 1 applied successfully.")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Apply patches then start the original Elena Router
    apply_patches()
    elena.run_agent()
