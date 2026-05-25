# Support Flow Bug Analysis: Phone Number in Support Ticket Flow

**Project:** Meallion Voice AI (LiveKit)
**Date:** 2026-05-25
**Files Analyzed:** `src/agents/el/agent.py`, `src/agents/en/agent.py`, `src/agents/el/tools.py`, `src/agents/en/tools.py`, `src/agents/tools/support_ticket.py`, `src/services/shopify.py`, `src/services/clickup.py`, `src/agents/patch1.py` → `patch7.py`

---

## Executive Summary

There are **two distinct but related bugs** in the support flow:

1. **Bug #1 – Agent still asks for a phone number during support ticket creation**, even though the intent was to remove the phone collection step from the ticket flow.
2. **Bug #2 – When the user provides a phone number, the agent immediately triggers a Shopify order lookup** instead of simply storing the number for later callback contact.

These bugs stem from **residual state machine code** and **architectural misalignment** between what was removed (LLM-facing tool signature) and what still exists (deterministic flow handler logic). The root cause spans multiple files and patch iterations.

---

## Part 1: Why the Agent Still Asks for a Phone Number

### 1.1 What Was Removed (Correctly)

The `create_support_ticket` **LLM-callable tool** in both `el/agent.py` and `en/agent.py` was correctly updated to **not require a phone number**:

```python
# el/agent.py — Lines 472–488 and en/agent.py — Lines 471–487
@llm.ai_callable()
async def create_support_ticket(
    self,
    customer_name: Annotated[str, llm.TypeInfo(description="Customer full name")],
    customer_email: Annotated[str, llm.TypeInfo(description="Customer email")],
    issue_description: Annotated[str, llm.TypeInfo(description="Issue")],
) -> str:
    """Create a support ticket... Only collect name, email, and issue. Do NOT ask for phone number."""
```

The docstring explicitly says **"Do NOT ask for phone number."**

Similarly, `_run_create_ticket()` in both agents calls `create_ticket_without_phone()` — a function that takes only name, email, and issue:

```python
# el/agent.py Lines 856–860 / en/agent.py Lines 853–857
result = await support_ticket.create_ticket_without_phone(
    customer_name=state.ticket_name or "Customer",
    customer_email=state.ticket_email,
    issue_description=state.ticket_issue,
)
```

> [!IMPORTANT]
> `create_ticket_without_phone` is called but **this function does not exist in `tools.py`** for either the `el` or `en` module. It was never added. This causes an `AttributeError` at runtime.

### 1.2 The Missing Function: `create_ticket_without_phone`

Search results confirm that **`create_ticket_without_phone` is only ever called, never defined** in the active codebase:

| File | Line | Usage |
|------|------|-------|
| `el/agent.py` | 481 | `await support_ticket.create_ticket_without_phone(...)` |
| `el/agent.py` | 856 | `await support_ticket.create_ticket_without_phone(...)` |
| `en/agent.py` | 480 | `await support_ticket.create_ticket_without_phone(...)` |
| `en/agent.py` | 853 | `await support_ticket.create_ticket_without_phone(...)` |

Both `el/tools.py` and `en/tools.py` define `create_support_ticket()` with a mandatory `customer_phone` parameter. **There is no `create_ticket_without_phone` function in either tools module**, meaning the agent will crash with `AttributeError` when trying to create a ticket.

### 1.3 The `SessionState` Still Has `ticket_phone`

Even though phone was "removed", the `SessionState` dataclass in both agents still has `ticket_phone` as a field:

```python
# el/agent.py Lines 319–348 / en/agent.py Lines 318–347
@dataclass
class SessionState:
    support_state: str = "idle"
    # ...
    ticket_phone: str = ""   # ← Still here, never populated now
```

The state machine comment even lists `ticket_phone` as a valid `support_state` value in the older patches but removed the state from the live flow — **without removing the field or the associated cleanup code**.

### 1.4 The `SessionState.support_state` Comment Still Shows `ticket_phone`

```python
# el/agent.py Line 321
support_state: str = "idle"  # idle|awaiting_order|checking_order|awaiting_phone|checking_phone|ticket_name|ticket_email|ticket_issue|ticket_confirm|creating_ticket
```

Notably, `ticket_phone` is **absent from this comment** in the latest `el/agent.py` and `en/agent.py`, confirming the state was intentionally removed. But the code that resets `ticket_phone` during cancellation is still present:

```python
# en/agent.py Lines 1374–1377 (ticket cancel)
state.ticket_name = ""
state.ticket_phone = ""   # ← Resetting a field that's never set
state.ticket_email = ""
state.ticket_issue = ""
```

### 1.5 The Old Patch Files Still Have `ticket_phone` State (Root Cause of Confusion)

The original behavior (asking for phone) is preserved in the archived patch files:

| File | Lines | Behavior |
|------|-------|---------|
| `patch1.py` | 1168–1181 | `support_state = "ticket_phone"` → asks for phone → stores in `state.ticket_phone` |
| `patch3.py` | 1166–1179 | Same pattern |
| `patch5.py` | 1244–1257 | Same pattern |
| `patch7.py` | 1244–1257 | Same pattern |

The **live** `el/agent.py` and `en/agent.py` skip the `ticket_phone` state entirely:

```
Old flow (patch1–patch7):
  ticket_name → ticket_phone → ticket_email → ticket_issue → ticket_confirm

New flow (el/agent.py, en/agent.py):
  ticket_name → ticket_email → ticket_issue → ticket_confirm
```

The phone step was correctly removed from the **flow**, but:
- `SessionState.ticket_phone` field was kept
- `create_ticket_without_phone()` was referenced but never created in `tools.py`
- Silence prompts for `awaiting_phone` state still mention "phone number for the order" — conflating two different phone purposes

### 1.6 The System Prompt Still Comes From the Database

Both agents load their prompts from the database at runtime via `prompts.py`:

```python
# el/prompts.py Lines 162–164
def build_system_prompt(language: str = "el") -> str:
    kb_content = load_knowledge_base(language)
    prompts_content = get_prompts_content(language)
```

**If the database still contains old prompts that instruct the LLM to ask for a phone number**, those instructions will override or conflict with the code-level changes. The system prompt is the **highest-priority instruction source** for the LLM, so even if the code removed the phone step, the LLM will still ask for a phone if the DB prompt says to.

> [!WARNING]
> The database prompt content (stored in `PromptsContent` table per language) must also be updated to reflect the removal of phone collection in the support ticket flow. This is not a code-only fix.

---

## Part 2: Why the Agent Triggers a Shopify Order Lookup When User Provides a Phone Number

### 2.1 The Support Flow State Machine and Phone Detection

When the support ticket flow reaches `ticket_name` state, the user provides their name. The state transitions to `ticket_email`. However, **the deterministic phone detection logic runs on EVERY user turn**, outside and before the ticket flow state handlers.

This is the critical bug. Here is the exact execution path:

```python
# en/agent.py Lines 1154–1165 (user_speech_committed handler)
all_digits = "".join(_extract_digit_parts(user_text))
if len(all_digits) >= 3:
    suppress_llm(5.0)          # ← LLM is suppressed as soon as digits detected
    room_log("EARLY_DIGIT_SUPPRESSION", digits=len(all_digits))
```

Then at line 1281 (en) / 1281 (el), this check fires **regardless of whether the user is in a ticket flow**:

```python
# en/agent.py Lines 1280–1303
# 3) Active phone-support flow
if state.support_state in {"awaiting_phone", "checking_phone"} or len(all_digits) >= 10:
    # ...
    phone = _normalize_phone_for_lookup(user_text)
    if phone:
        asyncio.create_task(_run_phone_lookup(agent, phone))  # ← Shopify lookup!
        return
```

### 2.2 The Dangerous `or len(all_digits) >= 10` Condition

The condition at line 1281 contains an **unconditional OR**:

```python
if state.support_state in {"awaiting_phone", "checking_phone"} or len(all_digits) >= 10:
```

This means: **If the user says anything that contains 10+ digits — at ANY point in the conversation — the phone lookup will be triggered**, regardless of what state the support flow is in.

In the ticket flow scenario:
- User is at `ticket_name` state → agent asks "What is your name?"
- User says "My name is Bhavya" — fine, no digits.
- Agent transitions to `ticket_email`.
- User provides email — fine.
- Agent transitions to `ticket_issue`.
- User is at `ticket_issue` state → provides issue: "My order 1234567890 has a problem."
- The text contains 10 digits (`1234567890`).
- `len(all_digits) >= 10` is `True`.
- `_run_phone_lookup(agent, "1234567890")` is called against Shopify.

Even worse, if the user is at `ticket_confirm` and says "yes, my contact is 6942633977" — those 10 digits trigger a Shopify order lookup.

### 2.3 The `_run_phone_lookup` Function Calls Shopify, Not a Storage Function

```python
# el/agent.py Lines 812–845 / en/agent.py Lines 809–842
async def _run_phone_lookup(agent: VoicePipelineAgent, phone_number: str):
    state: SessionState = _current["state"]
    # ...
    result = await asyncio.wait_for(
        order_lookup.lookup_order_by_phone(phone_number),  # ← Shopify call
        timeout=10.0
    )
```

And in `el/tools.py` Lines 129–169 / `en/tools.py` Lines 129–169:

```python
async def lookup_order_by_phone(phone: ...) -> str:
    shopify = get_shopify_service()
    cleaned = shopify.clean_phone_number(phone)
    # ...
    orders = await shopify.lookup_order_by_phone(cleaned)  # ← Queries Shopify API
    # ...
    return f"I found {count} orders for this phone number. {summary}"
```

**There is no "store phone for later contact" path.** Every phone number the agent receives is fed directly into the Shopify order lookup. The intent of collecting the phone number (to enable callback contact) was never implemented in the deterministic handler.

### 2.4 State Ordering Problem: Ticket Flow Checks Come After Phone Detection

The order of state checks in `_on_user_speech_committed` is:

```
1. Digit suppression (if >= 3 digits)
2. Farewell detection
3. Lookup inflight guard
4. Ticket creation escape (if user says "open ticket" etc.)
5. ← awaiting_order / checking_order handler
6. ← awaiting_phone / checking_phone OR 10+ digits handler  ← BUG IS HERE
7. ← ticket_name handler
8. ← ticket_email handler
9. ← ticket_issue handler
10. ← ticket_confirm handler
11. Support/ticket intent detection
```

Handlers **7–10** (the ticket flow states) come **after** handler **6** (the Shopify phone lookup). So when the user is in any ticket state and provides 10+ digits, the phone lookup handler fires at step 6 and `return`s — **the ticket flow handlers at steps 7–10 are never reached**.

---

## Part 3: Silence Prompts Make Things Worse

When the agent is in `awaiting_phone` or `checking_phone` state, the silence monitor says:

```python
# el/agent.py Lines 1476–1479
if support_state in {"awaiting_phone", "checking_phone"}:
    if phase == 0:
        return "Παρακαλώ δώστε μου το τηλέφωνο που χρησιμοποιήσατε για την παραγγελία όταν είστε έτοιμοι."
```

Translation: *"Please give me the phone number you used for the order when you are ready."*

This is correct for the order-lookup flow. However, this same silence prompt fires even if somehow the agent ends up in `awaiting_phone` during the support ticket flow, further confusing the user about why the agent is asking for a phone number "for the order."

---

## Part 4: The `create_ticket_without_phone` Missing Function

Both `el/tools.py` and `en/tools.py` only expose:

- `create_support_ticket(customer_name, customer_phone, customer_email, issue_description, order_number)` — **requires phone**
- `log_customer_query(...)` — quick logging
- `validate_ticket_field(...)` — validation helper

Neither module has `create_ticket_without_phone`. The agents call `support_ticket.create_ticket_without_phone(...)` which resolves to the module alias, and since `support_ticket` is aliased as `tools`:

```python
# el/agent.py Lines 37–39
from src.agents.el import tools as order_lookup
from src.agents.el import tools as knowledge_base
from src.agents.el import tools as support_ticket   # ← same module
```

Calling `support_ticket.create_ticket_without_phone(...)` will raise `AttributeError: module 'src.agents.el.tools' has no attribute 'create_ticket_without_phone'` at runtime.

---

## Part 5: Summary of All Root Causes

| # | Root Cause | Location | Impact |
|---|-----------|----------|--------|
| 1 | `create_ticket_without_phone` function called but never defined | `el/tools.py`, `en/tools.py` | `AttributeError` crash when ticket is created |
| 2 | `len(all_digits) >= 10` condition in phone flow check | `el/agent.py:1281`, `en/agent.py:1281` | Any 10-digit number triggers Shopify lookup in any state |
| 3 | Ticket flow state handlers (`ticket_name`, `ticket_email` etc.) come after phone detection handler | `el/agent.py:1280+`, `en/agent.py:1280+` | Phone lookup always preempts ticket flow |
| 4 | No "store phone for callback" path exists; all phone capture goes to Shopify lookup | `_run_phone_lookup()` in both agents | Wrong behavior — lookup instead of storing |
| 5 | `SessionState.ticket_phone` field kept but never populated | `el/agent.py:329`, `en/agent.py:328` | Dead code, potential future confusion |
| 6 | Database prompts may still instruct LLM to ask for phone | DB `PromptsContent` table | LLM behavior diverges from code intent |

---

## Part 6: Recommended Fixes

### Fix 1: Add `create_ticket_without_phone` to `tools.py` (Both `el` and `en`)

Add a wrapper function in `src/agents/el/tools.py` and `src/agents/en/tools.py`:

```python
async def create_ticket_without_phone(
    customer_name: str,
    customer_email: str,
    issue_description: str,
) -> dict:
    """
    Create a support ticket WITHOUT requiring phone number.
    Phone number is not collected — contact is via email only.
    Returns dict with 'message' key for the agent to speak.
    """
    result = await clickup_service.create_support_ticket(
        customer_name=customer_name.strip(),
        customer_phone="not_provided",  # or empty string if ClickUp allows
        customer_email=customer_email.strip(),
        issue_description=issue_description.strip(),
        order_number=None,
    )
    if result.get("success"):
        return {
            "message": (
                f"Your support ticket has been created. "
                f"Reference: {result['task_id']}. "
                "We will contact you via email soon."
            )
        }
    return {"message": "Sorry, I could not create the support ticket. Please try again."}
```

### Fix 2: Guard the Phone Lookup Against Ticket Flow States

In both `el/agent.py` and `en/agent.py`, change line 1281:

```python
# BEFORE (buggy):
if state.support_state in {"awaiting_phone", "checking_phone"} or len(all_digits) >= 10:

# AFTER (fixed):
_in_ticket_flow = state.support_state in {
    "ticket_name", "ticket_email", "ticket_issue", "ticket_confirm", "creating_ticket"
}
if not _in_ticket_flow and (
    state.support_state in {"awaiting_phone", "checking_phone"} or len(all_digits) >= 10
):
```

This ensures that when the user is in a ticket flow state, the phone lookup handler is skipped entirely.

### Fix 3: Remove `ticket_phone` from `SessionState` (Both Agents)

Remove the unused field to avoid confusion:

```python
# Remove this line from SessionState in both el/agent.py and en/agent.py:
ticket_phone: str = ""
```

Also remove the `state.ticket_phone = ""` reset lines in the ticket cancel handler.

### Fix 4: Update Database Prompts

Ensure the `PromptsContent` for both `el` and `en` languages in the database explicitly states:
- For support tickets: **collect only name, email, and issue description**
- **Do NOT ask for phone number** in the ticket flow
- Phone number collection only happens for order lookup fallback

### Fix 5: (Optional) Implement a Real Callback Phone Storage Path

If the business requirement is to collect a phone number **for callback purposes** (not order lookup), a separate function and state should be created:

```python
async def _store_callback_phone(agent: VoicePipelineAgent, phone_number: str):
    """Store phone number for callback — does NOT query Shopify."""
    state: SessionState = _current["state"]
    state.ticket_phone = phone_number
    state.last_phone_number = phone_number
    await agent.say(
        f"Got it. I have your number {phone_number}. "
        "Our team will call you back soon.",
        allow_interruptions=True
    )
    state.support_state = "idle"
```

This should be called instead of `_run_phone_lookup()` when in a support/ticket context.

---

## Part 7: File Reference Map

| File | Role | Key Issue |
|------|------|-----------|
| [`el/agent.py`](file:///c:/Users/bhavy/Downloads/livekit/src/agents/el/agent.py) | Greek agent entrypoint & state machine | Bug #2 at L1281; Bug #1 at L481, L856 |
| [`en/agent.py`](file:///c:/Users/bhavy/Downloads/livekit/src/agents/en/agent.py) | English agent entrypoint & state machine | Bug #2 at L1281; Bug #1 at L480, L853 |
| [`el/tools.py`](file:///c:/Users/bhavy/Downloads/livekit/src/agents/el/tools.py) | Tool functions for el agent | Missing `create_ticket_without_phone`; L129 has `lookup_order_by_phone` → Shopify |
| [`en/tools.py`](file:///c:/Users/bhavy/Downloads/livekit/src/agents/en/tools.py) | Tool functions for en agent | Same as above |
| [`tools/support_ticket.py`](file:///c:/Users/bhavy/Downloads/livekit/src/agents/tools/support_ticket.py) | Old tool module | `create_support_ticket` still requires phone at L147 |
| [`services/shopify.py`](file:///c:/Users/bhavy/Downloads/livekit/src/services/shopify.py) | Shopify API service | `lookup_order_by_phone()` queries Shopify — correct for order flow, wrong for support |
| [`el/prompts.py`](file:///c:/Users/bhavy/Downloads/livekit/src/agents/el/prompts.py) | Prompt loader for el agent | Loads DB prompts — DB content may instruct LLM to ask for phone |
| [`en/prompts.py`](file:///c:/Users/bhavy/Downloads/livekit/src/agents/en/prompts.py) | Prompt loader for en agent | Same as above |
| `patch1.py` – `patch7.py` | Historical patch versions | Show original `ticket_phone` state that was removed |

---

## Conclusion

The core problem is that the **phone number removal was only partially implemented**:
- The LLM-facing function signature was correctly updated (phone removed from `create_support_ticket`).
- The deterministic state machine was updated (phone state removed from ticket flow).
- **But:** `create_ticket_without_phone` was never created in `tools.py`.
- **And:** The `len(all_digits) >= 10` phone detection guard runs unconditionally across all states, so any phone-like number the user mentions **always** triggers a Shopify lookup, even during a support ticket conversation.

The phone number the user provides in the support flow is being treated as an **order lookup key** rather than **contact information for a callback**, because the architecture has no separate path for storing callback phone numbers — all phone handling routes through the Shopify order lookup pipeline.
