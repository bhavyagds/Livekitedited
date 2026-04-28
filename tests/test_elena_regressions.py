"""
Regression tests for Elena order/phone lookup flow helpers.

Uses stdlib unittest so it can run in constrained environments without pytest.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _install_elena_import_stubs():
    """Install minimal stub modules required to import src/agents/elena.py."""

    old_modules: dict[str, object] = {}

    def _set_module(name: str, module: types.ModuleType) -> None:
        if name in sys.modules:
            old_modules[name] = sys.modules[name]
        sys.modules[name] = module

    # --- livekit stubs ---
    livekit = types.ModuleType("livekit")
    livekit_agents = types.ModuleType("livekit.agents")
    livekit_pipeline = types.ModuleType("livekit.agents.pipeline")
    livekit_plugins = types.ModuleType("livekit.plugins")
    livekit_openai = types.ModuleType("livekit.plugins.openai")
    livekit_silero = types.ModuleType("livekit.plugins.silero")
    livekit_elevenlabs = types.ModuleType("livekit.plugins.elevenlabs")

    class _Dummy:
        pass

    class _AutoSubscribe:
        AUDIO_ONLY = "audio_only"

    class _FunctionContext:
        pass

    class _TypeInfo:
        def __init__(self, description: str = "") -> None:
            self.description = description

    class _ChatContext:
        def __init__(self) -> None:
            self.messages = []

        def append(self, **kwargs) -> None:
            self.messages.append(kwargs)

    def _ai_callable(*args, **kwargs):
        def _decorator(func):
            return func

        return _decorator

    llm_stub = types.SimpleNamespace(
        FunctionContext=_FunctionContext,
        TypeInfo=_TypeInfo,
        ChatContext=_ChatContext,
        ai_callable=_ai_callable,
    )

    livekit_agents.AutoSubscribe = _AutoSubscribe
    livekit_agents.JobContext = _Dummy
    livekit_agents.JobProcess = _Dummy
    livekit_agents.WorkerOptions = _Dummy
    livekit_agents.cli = types.SimpleNamespace(run_app=lambda *a, **k: None)
    livekit_agents.llm = llm_stub
    livekit_pipeline.VoicePipelineAgent = _Dummy

    _set_module("livekit", livekit)
    _set_module("livekit.agents", livekit_agents)
    _set_module("livekit.agents.pipeline", livekit_pipeline)
    _set_module("livekit.plugins", livekit_plugins)
    _set_module("livekit.plugins.openai", livekit_openai)
    _set_module("livekit.plugins.silero", livekit_silero)
    _set_module("livekit.plugins.elevenlabs", livekit_elevenlabs)

    # --- src.agents submodule stubs ---
    config_mod = types.ModuleType("src.config")
    config_mod.settings = types.SimpleNamespace(
        elevenlabs_api_key="",
        elevenlabs_model="eleven_multilingual_v2",
        elevenlabs_voice_id="voice-id",
        elevenlabs_voice_similarity=0.8,
    )
    _set_module("src.config", config_mod)

    src_agents_pkg = types.ModuleType("src.agents")
    src_agents_pkg.__path__ = []  # mark as package
    _set_module("src.agents", src_agents_pkg)

    energy_vad_mod = types.ModuleType("src.agents.energy_vad")
    energy_vad_mod.EnergyVAD = _Dummy
    _set_module("src.agents.energy_vad", energy_vad_mod)

    prompts_mod = types.ModuleType("src.agents.prompts")
    prompts_mod.get_system_prompt = lambda *a, **k: ""
    prompts_mod.get_system_prompt_async = lambda *a, **k: ""
    prompts_mod.get_greeting = lambda *a, **k: ""
    prompts_mod.get_closing = lambda *a, **k: ""
    prompts_mod.get_stt_language = lambda *a, **k: "en"
    prompts_mod.get_agent_language = lambda *a, **k: "en"
    prompts_mod.get_agent_setting = lambda key, default=None: default
    prompts_mod.set_runtime_language = lambda *a, **k: None
    _set_module("src.agents.prompts", prompts_mod)

    tools_pkg = types.ModuleType("src.agents.tools")
    tools_pkg.__path__ = []
    _set_module("src.agents.tools", tools_pkg)

    order_lookup_mod = types.ModuleType("src.agents.tools.order_lookup")

    async def _lookup_order(_order_number: str) -> str:
        return "Order 12345 is completed."

    async def _get_order_details(_order_number: str) -> str:
        return "ORDER DETAILS FOR #12345"

    async def _lookup_order_by_phone(_phone: str) -> str:
        return "I found one order for you."

    order_lookup_mod.lookup_order = _lookup_order
    order_lookup_mod.get_order_details = _get_order_details
    order_lookup_mod.lookup_order_by_phone = _lookup_order_by_phone
    order_lookup_mod.get_last_order_snapshot = lambda: {"order_number": "12345", "status": "completed"}
    order_lookup_mod.prefetch_orders = lambda: None
    _set_module("src.agents.tools.order_lookup", order_lookup_mod)

    support_ticket_mod = types.ModuleType("src.agents.tools.support_ticket")
    support_ticket_mod.create_support_ticket = lambda *a, **k: None
    support_ticket_mod.validate_ticket_field = lambda *a, **k: None
    support_ticket_mod.log_customer_query = lambda *a, **k: None
    _set_module("src.agents.tools.support_ticket", support_ticket_mod)

    kb_mod = types.ModuleType("src.agents.tools.knowledge_base")
    kb_mod.search_knowledge_base = lambda *a, **k: None
    kb_mod.get_brand_info = lambda *a, **k: None
    _set_module("src.agents.tools.knowledge_base", kb_mod)

    utils_mod = types.ModuleType("src.utils")
    utils_mod.detect_language = lambda text, default="en": default
    _set_module("src.utils", utils_mod)

    return old_modules


def _restore_modules(old_modules: dict[str, object]) -> None:
    for name in list(sys.modules.keys()):
        if (
            name.startswith("livekit")
            or name == "src.agents"
            or name.startswith("src.agents.")
            or name == "src.utils"
            or name == "src.config"
        ) and name not in old_modules:
            del sys.modules[name]
    for name, module in old_modules.items():
        sys.modules[name] = module


def _settings_getter_factory(overrides: dict):
    def _getter(key, default=None):
        return overrides.get(key, default)

    return _getter


class TestElenaRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._old_modules = _install_elena_import_stubs()
        module_path = Path("src/agents/elena.py").resolve()
        spec = importlib.util.spec_from_file_location("elena_regression_under_test", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.elena = module

    @classmethod
    def tearDownClass(cls) -> None:
        _restore_modules(cls._old_modules)

    def test_order_id_normalization_exact_digits(self):
        self.assertEqual(self.elena._normalize_order_id_strict("12345"), "12345")

    def test_order_id_normalization_spoken_digits(self):
        self.assertEqual(
            self.elena._normalize_order_id_strict("one two three four five"),
            "12345",
        )

    def test_order_id_normalization_rejects_wrong_length(self):
        self.assertIsNone(self.elena._normalize_order_id_strict("1234"))
        self.assertIsNone(self.elena._normalize_order_id_strict("12 34"))

    def test_phone_normalization_accepts_local_greek_mobile(self):
        self.assertEqual(
            self.elena._normalize_phone_for_lookup("69 1234 5678"),
            "6912345678",
        )

    def test_phone_normalization_accepts_prefixed_greek_mobile(self):
        self.assertEqual(
            self.elena._normalize_phone_for_lookup("0030 69 1234 5678"),
            "00306912345678",
        )
        self.assertEqual(
            self.elena._normalize_phone_for_lookup("30 69 1234 5678"),
            "306912345678",
        )

    def test_phone_normalization_rejects_invalid_numbers(self):
        self.assertIsNone(self.elena._normalize_phone_for_lookup("94269"))
        self.assertIsNone(self.elena._normalize_phone_for_lookup("69263977"))
        self.assertIsNone(self.elena._normalize_phone_for_lookup("1234567"))

    def test_phone_normalization_accepts_generic_full_numbers(self):
        self.assertEqual(self.elena._normalize_phone_for_lookup("9999999999"), "9999999999")
        self.assertEqual(self.elena._normalize_phone_for_lookup("2101234567"), "2101234567")

    def test_lookup_classifier_catches_common_not_found_variants(self):
        self.assertEqual(
            self.elena._classify_lookup_result("No matching order was found."),
            "not_found",
        )
        self.assertEqual(
            self.elena._classify_lookup_result("0 orders found for this phone."),
            "not_found",
        )
        self.assertEqual(
            self.elena._classify_lookup_result("Order not found."),
            "not_found",
        )

    def test_order_summary_not_found_is_deterministic(self):
        text = self.elena._build_order_voice_summary("No matching order was found.", "en")
        self.assertIn("I couldn't find that order.", text)

    def test_order_summary_unknown_has_safe_fallback(self):
        text = self.elena._build_order_voice_summary("System unavailable, try later.", "en")
        self.assertIn("I couldn't verify this order", text)

    def test_phone_summary_not_found_is_deterministic(self):
        text = self.elena._build_phone_lookup_voice_summary("No orders found for this phone.", "en")
        self.assertIn("I couldn't find any order with this phone number.", text)

    def test_phone_summary_unknown_has_safe_fallback(self):
        text = self.elena._build_phone_lookup_voice_summary("Backend timeout.", "en")
        self.assertIn("I couldn't verify any order with this phone number.", text)

    def test_speak_digits_uses_digit_by_digit_words(self):
        self.assertEqual(
            self.elena._speak_digits("69123", "en"),
            "six nine one two three",
        )
        self.assertEqual(
            self.elena._speak_digits("69123", "el"),
            "έξι εννέα ένα δύο τρία",
        )

    def test_lookup_classifier_marks_sparse_order_payload_as_found(self):
        payload = "Order #12345 Paid Delivery: 2026-04-30"
        self.assertEqual(self.elena._classify_lookup_result(payload), "found")

    def test_lookup_by_phone_hard_blocks_when_confirmation_pending(self):
        settings = {
            "order_lookup_wait_phrase_enabled": True,
            "invalid_number_recovery_silence_grace_seconds": 12.0,
        }
        self.elena._current_session["pending_phone_candidate"] = "6912345678"
        self.elena._current_session["support_flow_state"] = self.elena.FLOW_AWAITING_PHONE_CONFIRMATION
        self.elena._current_session["last_user_turn_id"] = 0
        self.elena._current_session["phone_forced_turn_id"] = 0
        self.elena._current_session["phone_forced_pending_turn_id"] = 0
        self.elena._current_session["phone_lookup_inflight"] = False
        self.elena._current_session["number_mode_lock"] = None
        self.elena._current_session["number_mode_turn_id"] = 0

        with patch.object(self.elena, "get_agent_setting", _settings_getter_factory(settings)), patch.object(
            self.elena, "get_agent_language", lambda: "en"
        ), patch.object(self.elena, "room_log", lambda *a, **k: None), patch.object(
            self.elena, "_snooze_silence_prompts", lambda *a, **k: None
        ):
            ctx = self.elena.ElenaFunctionContext()
            result = asyncio.run(ctx.lookup_order_by_phone("6912345678"))

        self.assertIn("please confirm whether this phone number is correct", result.lower())

    def test_lookup_order_does_not_embed_wait_phrase(self):
        settings = {
            "order_id_exact_digits": 5,
            "order_lookup_wait_phrase_enabled": True,
            "order_lookup_silence_grace_seconds": 8.0,
        }

        async def _fake_lookup(_order_number: str) -> str:
            return "Order 12345 is completed. Delivery on 2026-04-20. Total: 11.50"

        with patch.object(self.elena, "get_agent_setting", _settings_getter_factory(settings)), patch.object(
            self.elena, "get_agent_language", lambda: "en"
        ), patch.object(self.elena, "room_log", lambda *a, **k: None), patch.object(
            self.elena, "_snooze_silence_prompts", lambda *a, **k: None
        ), patch.object(
            self.elena.order_lookup, "lookup_order", _fake_lookup
        ):
            ctx = self.elena.ElenaFunctionContext()
            result = asyncio.run(ctx.lookup_order("12345"))

        pending_phrase = str(self.elena._current_session.get("pending_lookup_wait_phrase") or "")
        self.assertTrue(pending_phrase)
        self.assertNotIn(pending_phrase, result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
