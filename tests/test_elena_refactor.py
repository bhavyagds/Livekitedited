import unittest
from unittest.mock import MagicMock, patch
import sys

# Comprehensive mocks
MOCK_MODULES = [
    'livekit', 'livekit.agents', 'livekit.agents.pipeline', 'livekit.plugins', 
    'livekit.plugins.openai', 'livekit.plugins.silero', 'livekit.plugins.elevenlabs', 'livekit.rtc',
    'pydantic', 'pydantic_settings', 'sqlalchemy', 'sqlalchemy.ext', 'sqlalchemy.ext.asyncio',
    'sqlalchemy.orm', 'sqlalchemy.dialects', 'sqlalchemy.dialects.postgresql',
    'numpy', 'scipy', 'scipy.signal', 'openai', 'httpx', 'aiohttp',
    'src.services', 'src.models', 'src.agents.energy_vad', 'src.utils.language_detect',
    'src.agents.tools', 'src.config', 'src.utils.greek_numbers'
]
for mod in MOCK_MODULES:
    sys.modules[mod] = MagicMock()

from src.utils.tts_normalize import normalize_numeric_ids_for_tts
from src.agents.elena.helpers import (
    _extract_digit_parts, 
    _normalize_phone_for_lookup, 
    _normalize_order_id_strict
)
from src.agents.elena.state import _set_support_flow_state
from src.agents.elena.context import _current_session, FLOW_IDLE, FLOW_AWAITING_ORDER_NUMBER

class TestElenaRefactor(unittest.TestCase):
    def test_tts_normalization_digits(self):
        """Verify that phone numbers are converted to space-separated digits."""
        text_en = "Your phone number is 8003076358."
        normalized_en = normalize_numeric_ids_for_tts(text_en, language="en")
        self.assertIn("8 0 0 3 0 7 6 3 5 8", normalized_en)

    def test_extract_digit_parts(self):
        self.assertEqual(_extract_digit_parts("My number is 1 2 3"), ["1", "2", "3"])

    def test_phone_normalization(self):
        self.assertEqual(_normalize_phone_for_lookup("6940000000"), "+306940000000")

    def test_order_id_normalization(self):
        with patch('src.agents.prompts.get_agent_setting', return_value=5):
            self.assertEqual(_normalize_order_id_strict("Order 12345"), "12345")

    def test_state_transition(self):
        """Verify state transition logic and logging."""
        _current_session["support_flow_state"] = FLOW_IDLE
        with patch('src.agents.elena.state.room_log') as mock_log:
            _set_support_flow_state(FLOW_AWAITING_ORDER_NUMBER, reason="user_asking")
            self.assertEqual(_current_session["support_flow_state"], FLOW_AWAITING_ORDER_NUMBER)
            mock_log.assert_called_once_with(
                "SUPPORT_FLOW_STATE", 
                previous=FLOW_IDLE, 
                current=FLOW_AWAITING_ORDER_NUMBER, 
                reason="user_asking"
            )

if __name__ == "__main__":
    unittest.main()
