import unittest
import sys
import os
import re

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mocking some dependencies since we are testing logic functions
def mock_get_agent_setting(key, default):
    settings = {
        "order_id_exact_digits": 5,
        "phone_lookup_min_digits": 10,
    }
    return settings.get(key, default)

# Import the logic from elena.py (simulated for standalone test)
# In a real environment, you'd import directly from agents.elena
# For this test, we replicate the core logic we implemented.

def _normalize_order_id_strict(raw_text, min_len=4):
    normalized = (raw_text or "").strip().lower()
    if not normalized: return None
    
    # Check if it has any digits
    if not any(char.isdigit() for char in normalized):
        return None

    # Extract digits
    digits = "".join(re.findall(r"\d", normalized))
    
    # Validation: min 4, max 9 (to avoid phone)
    if len(digits) >= min_len and len(digits) < 10:
        return digits
    return None

def _speak_digits(digits, lang="en"):
    map_en = {"0":"zero","1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine"}
    map_el = {"0":"μηδέν","1":"ένα","2":"δύο","3":"τρία","4":"τέσσερα","5":"πέντε","6":"έξι","7":"επτά","8":"οκτώ","9":"εννέα"}
    
    mapping = map_el if lang == "el" else map_en
    return " ".join(mapping.get(d, d) for d in str(digits))

class TestElenaLogic(unittest.TestCase):

    def test_order_id_normalization(self):
        # Test minimum 4 digits
        self.assertEqual(_normalize_order_id_strict("1234"), "1234")
        self.assertEqual(_normalize_order_id_strict("12345"), "12345")
        self.assertEqual(_normalize_order_id_strict("1234567"), "1234567")
        
        # Test failure cases
        self.assertIsNone(_normalize_order_id_strict("123"))      # Too short
        self.assertIsNone(_normalize_order_id_strict("ABC"))      # No digits
        self.assertIsNone(_normalize_order_id_strict("6942633977")) # Too long (Phone)

    def test_digit_speech_formatting(self):
        # English
        self.assertEqual(_speak_digits("123", "en"), "one two three")
        self.assertEqual(_speak_digits("080", "en"), "zero eight zero")
        
        # Greek
        self.assertEqual(_speak_digits("123", "el"), "ένα δύο τρία")
        self.assertEqual(_speak_digits("0", "el"), "μηδέν")

    def test_transcript_formatting(self):
        # Simulated transcript formatter logic
        def format_transcript(text, pending_phone="6942633977"):
            lowered = text.lower()
            if "phone number" in lowered or "αριθμό τηλεφώνου" in lowered or "αριθμός τηλεφώνου" in lowered:
                return f"Phone confirmation: {pending_phone}. Is that correct?"
            return text

        self.assertIn("6942633977", format_transcript("Your phone number is zero eight zero..."))
        self.assertIn("6942633977", format_transcript("Ο αριθμός τηλεφώνου σας είναι..."))

    def test_silence_monitor_blocking(self):
        # Simulate silence monitor logic
        def should_prompt_silence(is_lookup_active, silence_elapsed, timeout=12.0):
            if is_lookup_active:
                return False # Always block during active search
            return silence_elapsed > timeout

        self.assertFalse(should_prompt_silence(is_lookup_active=True, silence_elapsed=15.0))
        self.assertTrue(should_prompt_silence(is_lookup_active=False, silence_elapsed=15.0))

    def test_periodic_lookup_updates(self):
        # Simulate the 15s progress update logic
        def needs_progress_update(last_update_at, current_time, interval=15.0):
            return (current_time - last_update_at) >= interval

        self.assertTrue(needs_progress_update(100, 115))  # Exactly 15s
        self.assertFalse(needs_progress_update(100, 110)) # Only 10s passed

    def test_memory_context_framing(self):
        # Simulate the behavioral guidelines framing logic
        def format_memory(items):
            header = "### Behavioral Guidelines and Historical Scenarios\n"
            body = "\n".join([f"- Scenario: {i['q']}\n  Guideline: {i['a']}" for i in items])
            return header + body

        memories = [{"q": "User is angry", "a": "Be extra polite"}]
        result = format_memory(memories)
        self.assertIn("Behavioral Guidelines", result)
        self.assertIn("Scenario: User is angry", result)

    def test_order_not_found_messaging(self):
        # Test localized "not found" messages
        def get_not_found_msg(lang):
            if lang == "el":
                return "Λυπάμαι, αλλά δεν μπόρεσα να βρω την παραγγελία σας"
            return "I'm sorry, but I couldn't find your order"

    def test_phone_fast_path_beats_invalid_order_digits(self):
        def normalize_phone_for_lookup(text):
            digits = "".join(re.findall(r"\d", text or ""))
            return digits if len(digits) == 10 and digits.startswith("69") else None

        def should_route_to_phone(flow_state, user_text, last_agent_text, number_mode_lock):
            if flow_state != "awaiting_order_number":
                return False
            if not re.search(r"\d", user_text or ""):
                return False

            normalized_phone = normalize_phone_for_lookup(user_text)
            is_phone_prompt = "τηλεφώνου" in (last_agent_text or "").lower()
            if normalized_phone and (is_phone_prompt or number_mode_lock == "phone"):
                return True

            return False

        self.assertTrue(
            should_route_to_phone(
                "awaiting_order_number",
                "6942633977",
                "Παρακαλώ πείτε τον αριθμό τηλεφώνου που χρησιμοποιήσατε για την παραγγελία, ψηφίο προς ψηφίο.",
                "phone",
            )
        )

    def test_phone_context_does_not_switch_to_order_on_casual_order_mention(self):
        def infer_number_mode(user_text, last_agent_text, phone_flow_active=True):
            lowered = re.sub(r"[^\w\s]", " ", (user_text or "").lower())
            lowered = re.sub(r"\s+", " ", lowered).strip()
            last_lowered = re.sub(r"[^\w\s]", " ", (last_agent_text or "").lower())
            last_lowered = re.sub(r"\s+", " ", last_lowered).strip()

            order_hint = bool(re.search(r"(order|order id|order number)", lowered))
            phone_hint = bool(re.search(r"(phone|mobile)", lowered))
            asked_for_phone = bool(re.search(r"(phone|mobile)", last_lowered))
            has_digits = bool(re.search(r"\d", lowered))

            order_switch_patterns = (
                r"\b(?:use|using|search|lookup|check|try)\b.*\b(order|order id|order number)\b",
                r"\b(order|order id|order number)\b.*\b(?:instead|please|now)\b",
                r"\b(?:search by|find by|lookup by|check by)\b.*\b(order|order id|order number)\b",
            )
            if order_hint and not phone_hint and any(re.search(pattern, lowered) for pattern in order_switch_patterns):
                return "order"
            if phone_flow_active:
                if phone_hint or asked_for_phone or has_digits:
                    return "phone"
                if order_hint:
                    return None
            if phone_hint and not order_hint:
                return "phone"
            if order_hint and not phone_hint:
                return "order"
            return None

        self.assertEqual(
            infer_number_mode(
                "oh please find what order id",
                "Please give me the phone number used for the order, digit by digit.",
            ),
            "phone",
        )
        self.assertEqual(
            infer_number_mode(
                "please search with order id instead",
                "Please give me the phone number used for the order, digit by digit.",
            ),
            "order",
        )

    def test_yes_counts_as_details_request_when_prompt_was_just_spoken(self):
        def explicit_more_order_details_request(text, prompted_at, now):
            lowered = re.sub(r"[^\w\s]", " ", (text or "").lower())
            lowered = re.sub(r"\s+", " ", lowered).strip()
            yes_tokens = {"yes", "yeah", "yep", "sure", "ok", "okay"}
            if lowered in yes_tokens and prompted_at and (now - prompted_at) <= 25.0:
                return True
            return False

        self.assertTrue(explicit_more_order_details_request("Yes", 100.0, 103.0))
        self.assertFalse(explicit_more_order_details_request("Yes", 100.0, 140.0))

    def test_memory_context_formatting_v2(self):
        # Exact replication of database.py:get_active_memory_context formatting logic
        def build_memory_context(memories):
            if not memories: return ""
            sections = []
            sections.append("### BEHAVIORAL GUIDELINES & EXAMPLES")
            sections.append("The following examples illustrate how you should behave and answer in specific situations. Learn from these patterns to provide consistent service:")
            sections.append("")
            for m in memories:
                q, a, c = m.get("q"), m.get("a"), m.get("c")
                sections.append(f"SCENARIO: User asks/says: \"{q}\"")
                sections.append(f"EXPECTED RESPONSE: \"{a}\"")
                if c: sections.append(f"GUIDELINE: {c}")
                sections.append("-" * 20)
            return "\n".join(sections).strip()

        # Test with multiple memories and comments
        test_data = [
            {"q": "How do I return?", "a": "Check our portal.", "c": "Always be helpful."},
            {"q": "Missing item", "a": "I will check for you.", "c": "Use empathetic tone."}
        ]
        result = build_memory_context(test_data)
        
        self.assertIn("BEHAVIORAL GUIDELINES & EXAMPLES", result)
        self.assertIn("SCENARIO: User asks/says: \"How do I return?\"", result)
        self.assertIn("EXPECTED RESPONSE: \"I will check for you.\"", result)
        self.assertIn("GUIDELINE: Use empathetic tone.", result)
        self.assertEqual(result.count("-" * 20), 2)

if __name__ == "__main__":
    unittest.main()
