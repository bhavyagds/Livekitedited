"""
Meallion Voice AI - Elena Voice Agent
Main voice agent implementation using LiveKit Agents SDK (2026 version).
"""

import logging
import asyncio
import time
import os
import re
import json
import random
import threading
import unicodedata
from datetime import datetime
from typing import Annotated, Optional

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, silero, elevenlabs

# Try to use Deepgram for faster STT (optional)
try:
    from livekit.plugins import deepgram
    USE_DEEPGRAM = True
except ImportError:
    USE_DEEPGRAM = False

from src.config import settings
from src.agents.energy_vad import EnergyVAD
from src.agents.prompts import (
    get_system_prompt, get_system_prompt_async, get_greeting, get_closing, get_stt_language,
    get_agent_language, get_agent_setting, set_runtime_language
)
from src.agents.tools import order_lookup, support_ticket, knowledge_base
from src.utils import detect_language

logger = logging.getLogger(__name__)



from .context import _current_session
from .logger import LatencyTracker, log_call_event, record_call_to_db, end_call_in_db, _create_room_logger, room_log
from .state import *
from .helpers import *
from .providers import create_llm, create_tts, create_stt, create_vad
from .tools import ElenaFunctionContext

async def create_initial_context(cache_task: asyncio.Task = None) -> llm.ChatContext:
    """Create the initial chat context with system prompt based on configured language.
    
    Args:
        cache_task: Optional task from _fetch_from_db() to wait for before getting prompts.
                   This ensures the cache is ready without re-fetching.
    """
    # Wait for cache to be ready if a task was provided
    if cache_task is not None:
        try:
            await cache_task
        except Exception as e:
            logger.warning(f"Cache task exception (will retry): {e}")
    
    ctx = llm.ChatContext()
    agent_lang = get_agent_language()
    
    # Use async version to ensure KB and prompts are loaded from DB
    system_prompt = await get_system_prompt_async(agent_lang)
    
    logger.info(f"Using {agent_lang} system prompt (from database), length: {len(system_prompt)} chars")
    ctx.append(role="system", text=system_prompt)
    return ctx


async def entrypoint(ctx: JobContext):
    """LiveKit Agent entrypoint. Called when a new participant joins the room."""
    global _current_session
    
    startup_time = time.time()
    logger.info(f"Elena agent starting for room: {ctx.room.name}")
    logger.warning("VOICE_AGENT_VERSION: phone-confirmation-hard-gate-2026-04-28-v4")
    
    # Track call timing for metrics
    call_start_time = time.time()
    caller_number = None
    caller_identity = None
    db_call_id = None
    
    # Determine call type (SIP or Web)
    is_sip_call = ctx.room.name.startswith("sip-") or "sip" in ctx.room.name.lower()
    call_type = "sip" if is_sip_call else "web"
    
    # Extract caller info from room name if available (e.g., sip-call-+30211234567)
    if is_sip_call:
        parts = ctx.room.name.split("-")
        if len(parts) >= 3:
            caller_number = parts[-1] if parts[-1].startswith("+") else None

    # Create per-room log file (full lifecycle)
    job_obj = getattr(ctx, "job", None)
    job_id = getattr(job_obj, "id", None) or getattr(job_obj, "job_id", None)
    room_logger, room_log_path = _create_room_logger(ctx.room.name, job_id)
    _current_session["room_logger"] = room_logger
    _current_session["room_name"] = ctx.room.name
    _current_session["job_id"] = job_id
    room_log("ROOM_START", call_type=call_type, caller_number=caller_number)
    logger.info(f"Per-room log file: {room_log_path}")
    _reset_support_session_state("new_call_started")
    
    # Reset session state
    _current_session["should_end"] = False
    _current_session["silence_tracker"] = None
    _current_session["silence_pause_depth"] = 0
    _current_session["last_silence_block_reason"] = None
    _current_session["last_lookup_wait_phrase"] = None
    _current_session["pending_lookup_wait_phrase"] = None
    _current_session["pending_lookup_wait_phrase_set_at"] = 0.0
    _current_session["last_agent_text"] = ""
    _current_session["last_forced_lookup_order"] = None
    _current_session["last_forced_lookup_at"] = 0.0
    _current_session["last_user_turn_id"] = 0
    _current_session["last_lookup_tool_called_at"] = 0.0
    _current_session["last_lookup_tool_order"] = None
    _current_session["lookup_progress_prompt_until"] = 0.0
    _current_session["number_mode_lock"] = None
    _current_session["number_mode_turn_id"] = 0
    _current_session["forced_response_manual_say_active"] = False
    _current_session["forced_response_spoken_turn_id"] = 0
    _current_session["forced_response_spoken_text"] = ""
    _current_session["forced_response_suppress_llm_until"] = 0.0
    _current_session["lookup_pending"] = False
    _current_session["lookup_pending_started_at"] = 0.0
    _current_session["lookup_pending_order"] = None
    _current_session["last_lookup_state"] = "unknown"
    _current_session["last_lookup_order"] = None
    _current_session["pending_phone_candidate"] = None
    _current_session["phone_digit_buffer"] = ""
    _current_session["phone_digit_buffer_updated_at"] = 0.0
    _current_session["phone_lookup_inflight"] = False
    _current_session["phone_forced_turn_id"] = 0
    _current_session["phone_forced_pending_turn_id"] = 0
    _current_session["details_lookup_inflight"] = False
    _current_session["details_forced_turn_id"] = 0
    _current_session["details_forced_pending_turn_id"] = 0
    _current_session["details_last_spoken_order"] = None
    _current_session["details_last_spoken_at"] = 0.0
    _current_session["details_confirmation_pending"] = False
    _current_session["details_confirmation_pending_until"] = 0.0
    _current_session["full_order_details_allowed_until"] = 0.0
    _current_session["full_details_unlocked_once"] = False
    _current_session["last_more_details_prompt_at"] = 0.0
    _current_session["ticket_confirmation_pending"] = False
    _current_session["ticket_confirmation_pending_until"] = 0.0
    _current_session["ticket_create_allowed_until"] = 0.0
    _current_session["pending_ticket_payload"] = None
    _current_session["ticket_created"] = False
    _current_session["ticket_reference"] = None
    _current_session["support_flow_state"] = FLOW_IDLE
    _current_session["support_issue"] = None
    
    # =========================================================================
    # PARALLEL STARTUP - Run ALL independent operations concurrently for speed
    # Key insight: DB fetch is slowest (~10s), so start it first and run
    # everything else in parallel
    # =========================================================================
    
    # 1. Start cache refresh FIRST (this is the slowest operation - ~10s)
    from src.agents.prompts import _fetch_from_db
    cache_task = asyncio.create_task(_fetch_from_db())
    
    # 2. Start context creation immediately - it will wait for cache internally
    # This runs in parallel with room connection
    context_task = asyncio.create_task(create_initial_context(cache_task))
    
    # 3. Start other background tasks (fire and forget - don't wait for these)
    asyncio.create_task(log_call_event(
        event_type="call_incoming",
        room_name=ctx.room.name,
        call_type=call_type,
        caller_number=caller_number,
        metadata={"source": "livekit_agent"},
    ))
    
    # 4. Connect to room (runs in parallel with cache fetch + context creation)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"⏱️ Connected to room ({time.time() - startup_time:.1f}s)")
    room_log("ROOM_CONNECTED", elapsed_s=round(time.time() - startup_time, 1))
    
    # 5. Wait for participant (user connects) - runs in parallel with cache fetch
    participant = await ctx.wait_for_participant()
    logger.info(f"⏱️ Participant connected: {participant.identity} ({time.time() - startup_time:.1f}s)")
    room_log("PARTICIPANT_CONNECTED", identity=participant.identity, elapsed_s=round(time.time() - startup_time, 1))
    caller_identity = participant.identity
    
    # Try to get caller number from participant identity for SIP calls
    if is_sip_call and not caller_number:
        caller_number = participant.identity if participant.identity.startswith("+") else None
    
    # Record call start in background (don't block greeting)
    async def record_call_async():
        nonlocal db_call_id
        db_call_id = await record_call_to_db(
            room_name=ctx.room.name,
            call_type=call_type,
            caller_number=caller_number,
            caller_identity=caller_identity,
        )
        await log_call_event(
            event_type="call_connected",
            room_name=ctx.room.name,
            call_type=call_type,
            caller_number=caller_number,
            caller_identity=caller_identity,
            call_id=db_call_id,
            metadata={"source": "livekit_agent"},
        )
        logger.info(f"Call recorded in DB with ID: {db_call_id}")
        _current_session["call_id"] = db_call_id
        room_log("CALL_RECORDED", db_call_id=db_call_id)
    
    asyncio.create_task(record_call_async())
    
    # Initialize transcript collection
    conversation_transcript = []
    # Track if we already handled call end to avoid duplicate processing and late async sends.
    call_ended = {"value": False}
    
    # Context creation was started earlier (line ~520) in parallel with room connection
    logger.info(f"⏱️ Context creation in progress ({time.time() - startup_time:.1f}s)")
    
    # Track sent payloads to avoid duplicate UI updates.
    _sent_agent_transcripts = set()
    _sent_agent_info_payloads = set()
    _last_user_interim = ""
    _last_user_interim_sent_at = 0.0
    _last_user_final = ""
    
    # Ensure settings cache is ready before requiring DB-backed settings.
    if cache_task is not None:
        try:
            await cache_task
        except Exception as e:
            logger.warning(f"Cache task exception (settings): {e}")

    # Initialize runtime language (per-call) from DB defaults.
    set_runtime_language(None)
    base_language = get_agent_language()
    set_runtime_language(base_language)
    session_language = {"value": base_language}
    auto_language_switch = _as_bool(get_agent_setting("auto_language_switch", False), default=False)
    language_switch_min_turns_setting = get_agent_setting("language_switch_min_turns", 2)
    if is_sip_call:
        language_switch_min_turns_setting = get_agent_setting(
            "sip_language_switch_min_turns",
            language_switch_min_turns_setting,
        )
    language_switch_min_turns = _as_int(
        language_switch_min_turns_setting,
        2,
        min_value=1,
        max_value=5,
    )
    language_switch_state = {"candidate": None, "count": 0}
    language_lock_state = {"el": 0, "en": 0, "el_last": "", "en_last": ""}
    language_lock_cache: dict[tuple[str, str], str] = {}

    def _normalize_switch_text(text: str) -> str:
        """Normalize text for robust language-switch intent detection."""
        lowered = (text or "").strip().lower()
        if not lowered:
            return ""
        # Keep letters/numbers/spaces only to make phrase matching resilient.
        lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _explicit_language_request(text: str) -> Optional[str]:
        """Detect explicit caller requests to change response language."""
        lowered = _normalize_switch_text(text)
        if not lowered:
            return None

        english_requests = (
            "speak english",
            "speak in english",
            "can you speak english",
            "can you speak in english",
            "can we speak english",
            "can we speak in english",
            "in english",
            "english please",
            "switch to english",
            "switch language to english",
            "talk in english",
            "reply in english",
            "respond in english",
            "answer in english",
            "details in english",
            "english language",
            "mila agglika",
            "sta agglika",
            "μιλα αγγλικα",
            "στα αγγλικα",
            "μίλα αγγλικά",
            "στα αγγλικά",
        )
        greek_requests = (
            "speak greek",
            "speak in greek",
            "can you speak greek",
            "can you speak in greek",
            "can we speak greek",
            "can we speak in greek",
            "in greek",
            "greek please",
            "switch to greek",
            "switch language to greek",
            "talk in greek",
            "reply in greek",
            "respond in greek",
            "answer in greek",
            "greek language",
            "mila ellinika",
            "sta ellinika",
            "μιλα ελληνικα",
            "στα ελληνικα",
            "μίλα ελληνικά",
            "στα ελληνικά",
        )

        if any(phrase in lowered for phrase in english_requests):
            return "en"
        if any(phrase in lowered for phrase in greek_requests):
            return "el"

        # Fallback intent heuristics for broader phrasing coverage.
        english_token = ("english" in lowered) or ("αγγλικ" in lowered)
        greek_token = ("greek" in lowered) or ("ελληνικ" in lowered) or ("ellinik" in lowered)
        switch_verbs = (
            "speak", "talk", "reply", "respond", "answer", "switch", "language",
            "μιλα", "μίλα", "μιλησουμε", "μιλήσουμε", "απάντηση",
        )
        has_switch_verb = any(v in lowered for v in switch_verbs)
        if english_token and has_switch_verb:
            return "en"
        if greek_token and has_switch_verb:
            return "el"
        return None

    def _explicit_more_order_details_request(text: str) -> bool:
        """Detect when caller explicitly asks for full order details."""
        lowered = _normalize_switch_text(text)
        if not lowered:
            return False

        en_phrases = (
            "more details",
            "full details",
            "complete details",
            "show details",
            "order details",
            "item details",
            "what are the items",
            "what did i order",
            "details please",
        )
        el_phrases = (
            "περισσότερες λεπτομέρειες",
            "όλες τις λεπτομέρειες",
            "αναλυτικές λεπτομέρειες",
            "λεπτομέρειες παραγγελίας",
            "τι περιέχει η παραγγελία",
            "ποια είδη",
        )
        if any(p in lowered for p in en_phrases):
            return True
        if any(p in lowered for p in el_phrases):
            return True

        yes_tokens = {"yes", "yeah", "yep", "sure", "ok", "okay", "ναι", "εντάξει"}
        if lowered in yes_tokens:
            prompted_at = float(_current_session.get("last_more_details_prompt_at") or 0.0)
            if prompted_at and (time.time() - prompted_at) <= 25.0:
                return True
        return False

    # Reuse module-level numeric parsers to avoid divergence between scopes.
    _digits_from_phrase = globals()["_digits_from_phrase"]
    _extract_digit_parts = globals()["_extract_digit_parts"]
    _normalize_order_id_strict = globals()["_normalize_order_id_strict"]
    _normalize_phone_for_lookup = globals()["_normalize_phone_for_lookup"]

    def _format_user_text_for_transcript(raw_text: str) -> str:
        """
        Normalize number-heavy utterances to raw digits for saved transcripts.
        Avoid "[digits: ...]" annotations and keep transcript clean.
        """
        text = (raw_text or "").strip()
        if not text:
            return text

        phone_candidate = _normalize_phone_for_lookup(text)
        if phone_candidate:
            return phone_candidate

        order_candidate = _normalize_order_id_strict(text)
        if order_candidate:
            return order_candidate

        parts = _extract_digit_parts(text)
        joined = "".join(parts)
        if len(joined) >= 4 and _is_digit_collection_utterance(text):
            return joined

        return text

    def _format_agent_text_for_transcript(raw_text: str) -> str:
        """
        Normalize agent phone-confirmation lines to show numeric digits in transcripts.
        This affects transcript/log text only, not tool inputs.
        """
        text = (raw_text or "").strip()
        if not text:
            return text
        if get_agent_language() != "el":
            return text

        if not _is_phone_confirmation_prompt(text):
            return text

        pending_phone = str(_current_session.get("pending_phone_candidate") or "").strip()
        if pending_phone and re.fullmatch(r"\d{7,15}", pending_phone):
            return f"Phone confirmation: {pending_phone}. Is that correct?"

        digits = "".join(_extract_digit_parts(text))
        if len(digits) < 4:
            digits = ""

        if digits:
            return f"Phone confirmation: {digits}. Is that correct?"

        return text

    def _mentions_no_order_number(text: str) -> bool:
        """Detect caller intent that they do not have an order number."""
        lowered = _normalize_switch_text(text)
        if not lowered:
            return False
        return bool(
            re.search(
                r"(no order number|don t have (an )?order number|do not have (an )?order number|δεν έχω .*παραγγελ|δεν εχω .*παραγγελ)",
                lowered,
                flags=re.IGNORECASE,
            )
        )

    def _infer_number_mode(user_text: str, last_agent_text: str) -> Optional[str]:
        """
        Infer whether current numeric capture should be treated as order id or phone.
        Returns 'order', 'phone', or None.
        """
        lowered = _normalize_switch_text(user_text)
        last_lowered = _normalize_switch_text(last_agent_text)
        if not lowered:
            return None

        force_phone_context = _is_phone_flow_active()
        if _mentions_no_order_number(lowered):
            return "phone"

        order_hint = bool(re.search(r"(order|παραγγελ|αριθμ.*παραγγελ)", lowered))
        phone_hint = bool(re.search(r"(phone|mobile|τηλέφων|τηλεφων|κινητ)", lowered))

        asked_for_order = bool(re.search(r"(order number|παραγγελ|αριθμ.*παραγγελ)", last_lowered))
        asked_for_phone = bool(re.search(r"(phone|mobile|τηλέφων|τηλεφων|κινητ)", last_lowered))

        if force_phone_context and not order_hint:
            if phone_hint or asked_for_phone or bool(_extract_digit_parts(lowered)):
                return "phone"

        if phone_hint and not order_hint:
            return "phone"
        if order_hint and not phone_hint:
            return "order"
        if asked_for_phone and asked_for_order and bool(_extract_digit_parts(lowered)):
            return "phone"
        if asked_for_phone and not asked_for_order:
            return "phone"
        if asked_for_order and not asked_for_phone:
            return "order"
        return None

    def _is_digit_collection_utterance(text: str) -> bool:
        lowered = (text or "").lower().strip()
        if not lowered:
            return False
        parts = _extract_digit_parts(lowered)
        if len(parts) >= 3:
            return True
        return bool(re.search(r"(digit by digit|ψηφίο προς ψηφίο|ένα, δύο|one two)", lowered))

    def _is_short_utterance(text: str) -> bool:
        lowered = _normalize_switch_text(text)
        if not lowered:
            return True
        words = [w for w in lowered.split() if w]
        return len(words) <= 2

    def _is_phone_confirmation_prompt(text: str) -> bool:
        """Detect prompts that ask the caller to confirm a captured phone number."""
        lowered = _normalize_switch_text(text)
        if not lowered:
            return False
        strict_patterns = (
            r"just to confirm,?\s*(?:your )?(?:phone|mobile)(?: number)?\s+is",
            r"is this (?:your )?(?:phone|mobile)(?: number)? correct",
            r"did i get your (?:phone|mobile)(?: number)? right",
            r"can you confirm (?:this|your) (?:phone|mobile)(?: number)?",
            r"(?:επιβεβαι|επιβεβαιώ).*(?:κινητ|τηλέφων)",
            r"(?:τηλέφων|κινητ).*(?:σωστό|σωστα)",
            r"ξανα πειτε το κινητο",
            r"ξανά πείτε το κινητό",
        )
        return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in strict_patterns)

    def _can_unlock_full_details(user_text: str) -> tuple[bool, str]:
        """Allow full details only when we have a found lookup for the same order."""
        if bool(_current_session.get("full_details_unlocked_once")):
            return False, "already_unlocked_once"
        state = str(_current_session.get("last_lookup_state") or "unknown")
        if state != "found":
            return False, "lookup_not_found"
        anchor = re.sub(r"\D", "", str(_current_session.get("last_lookup_order") or ""))
        expected = _expected_order_digits()
        if not re.fullmatch(rf"\d{{{expected}}}", anchor):
            return False, "missing_anchor_order"
        mentioned = _normalize_order_id_strict(user_text)
        if mentioned and mentioned != anchor:
            return False, "order_mismatch"
        return True, "ok"

    def _is_clarification_prompt_text(text: str) -> bool:
        """Detect clarification/repair prompts so silence grace can be extended."""
        lowered = _normalize_switch_text(text)
        if not lowered:
            return False
        return bool(
            re.search(
                r"(repeat|say it again|digit by digit|not clear|couldn t hear|didn t hear|did not hear|could not hear|δεν .*άκουσα|δεν .*κατάλαβ|επαναλάβ|ξανά|ψηφίο προς ψηφίο)",
                lowered,
                flags=re.IGNORECASE,
            )
        )

    def _extract_order_number_candidate(text: str) -> Optional[str]:
        """
        Extract likely order number with strict exact configured digit length.
        Prioritizes chunks near order-related keywords to avoid false captures.
        """
        normalized = (text or "").lower().strip()
        if not normalized:
            return None
        expected = _expected_order_digits()

        explicit_runs = re.findall(rf"\d{{{expected}}}", normalized)
        if explicit_runs:
            return explicit_runs[-1]

        windows = []
        for match in re.finditer(r"(order(?:\s+number)?|παραγγε\w*|αριθμ\w*)", normalized, flags=re.IGNORECASE):
            windows.append(normalized[match.start(): match.start() + 96])

        if windows:
            for segment in windows:
                candidate = _normalize_order_id_strict(segment)
                if candidate and len(candidate) == expected:
                    return candidate

        fallback = _normalize_order_id_strict(normalized)
        if fallback and len(fallback) == expected:
            return fallback
        return None

    def _should_force_order_lookup(user_text: str, order_number: Optional[str]) -> bool:
        """Return True when we should force immediate lookup_order for this turn."""
        if not order_number:
            return False

        normalized_user = _normalize_switch_text(user_text)
        if re.search(r"(order|παραγγελ|αριθμ)", normalized_user, flags=re.IGNORECASE):
            return True

        last_agent = _normalize_switch_text(str(_current_session.get("last_agent_text") or ""))
        return bool(
            re.search(
                r"(order number|number from your confirmation|παραγγελ|αριθμ)",
                last_agent,
                flags=re.IGNORECASE,
            )
        )

    def _english_switch_confident(text: str) -> bool:
        """
        Return True only when transcript strongly looks like real English.
        This prevents Greek speech transliterated into latin characters
        from incorrectly switching the call to English.
        """
        raw = (text or "").strip()
        if not raw:
            return False

        # If any Greek script exists, do not treat as English auto-switch.
        if re.search(r"[\u0370-\u03FF\u1F00-\u1FFF]", raw):
            return False

        lowered = raw.lower()
        tokens = re.findall(r"[a-z']+", lowered)
        if len(tokens) < 4:
            return False

        # Function/content words that appear frequently in real English utterances.
        english_markers = {
            "i", "you", "we", "they", "he", "she", "it",
            "am", "is", "are", "was", "were",
            "have", "has", "had", "do", "did", "can",
            "the", "a", "an", "to", "for", "of", "in", "on", "with", "and",
            "please", "my", "your", "order", "problem", "food",
        }
        marker_hits = sum(1 for t in tokens if t in english_markers)
        marker_ratio = marker_hits / max(len(tokens), 1)

        # Be strict from Greek -> English: require clear English signal.
        return marker_hits >= 2 and marker_ratio >= 0.22

    def _allow_auto_language_switch(current_lang: str, detected_lang: str, text: str) -> bool:
        """Gate automatic switching to reduce false positives from noisy STT output."""
        if detected_lang == current_lang:
            return False
        if current_lang == "el" and detected_lang == "en":
            allowed = _english_switch_confident(text)
            if not allowed:
                room_log(
                    "LANGUAGE_SWITCH_SUPPRESSED",
                    current=current_lang,
                    candidate=detected_lang,
                    reason="low_english_confidence",
                )
            return allowed
        return True

    def _apply_language_switch(new_lang: str, reason: str) -> None:
        """Switch runtime/session language and append a scoped system hint."""
        session_language["value"] = new_lang
        set_runtime_language(new_lang)
        language_switch_state["candidate"] = None
        language_switch_state["count"] = 0
        lang_name = "Greek" if new_lang == "el" else "English"
        agent.chat_ctx.append(
            role="system",
            text=(
                "LANGUAGE SWITCH:\n"
                f"- Respond in {lang_name} for this response and until the caller switches again."
            ),
        )
        room_log("LANGUAGE_SWITCH", language=new_lang, reason=reason)

    def _get_language_lock_phrase(expected_lang: str, source_text: str) -> str:
        """Rotate lock phrases but keep deterministic output for the same source text."""
        phrases = {
            "el": [
                "Παρακαλώ, μπορείτε να συνεχίσετε στα ελληνικά; Ευχαριστώ!",
                "Συνεχίζουμε στα ελληνικά. Πείτε μου πώς μπορώ να βοηθήσω.",
                "Θα συνεχίσω στα ελληνικά. Πείτε μου το αίτημά σας.",
                "Μπορούμε να προχωρήσουμε στα ελληνικά. Πώς μπορώ να σας εξυπηρετήσω;",
            ],
            "en": [
                "Please continue in English. Thank you!",
                "We will continue in English. Tell me how I can help.",
                "I will continue in English. Please share your request.",
                "Let's proceed in English. How can I assist you?",
            ],
        }
        options = phrases.get(expected_lang) or phrases["en"]
        cache_key = (expected_lang, (source_text or "").strip())
        cached = language_lock_cache.get(cache_key)
        if cached:
            return cached

        idx = int(language_lock_state.get(expected_lang, 0)) % len(options)
        selected = options[idx]
        last_key = f"{expected_lang}_last"
        if len(options) > 1 and selected == language_lock_state.get(last_key):
            idx = (idx + 1) % len(options)
            selected = options[idx]

        language_lock_state[expected_lang] = idx + 1
        language_lock_state[last_key] = selected
        language_lock_cache[cache_key] = selected
        if len(language_lock_cache) > 200:
            language_lock_cache.clear()
        return selected

    def _enforce_locked_output_language(text: str) -> str:
        """
        Keep agent output in the admin-selected language when auto switching is disabled.
        This is a hard guardrail for noisy user requests like "reply in English".
        """
        if not text:
            return text
        if auto_language_switch:
            return text

        expected_lang = session_language["value"]
        detected_lang = detect_language(text, default=expected_lang)
        if detected_lang == expected_lang:
            return text

        room_log(
            "LANGUAGE_OUTPUT_NORMALIZED",
            expected=expected_lang,
            detected=detected_lang,
            reason="auto_switch_disabled",
        )
        return _get_language_lock_phrase(expected_lang, text)

    # Initialize abuse tracker for this session
    from src.utils.abuse_handler import AbuseTracker, check_and_respond_to_abuse
    _abuse_tracker = AbuseTracker()
    abuse_detection_enabled = _require_bool_setting("abuse_detection_enabled")
    
    # =========================================================================
    # SILENCE DETECTION - Prompt user if no response (2 prompts then disconnect)
    # =========================================================================
    silence_timeout_raw = get_agent_setting("silence_timeout_seconds", 12.0)
    max_prompts_raw = get_agent_setting("silence_max_prompts", 3)
    try:
        silence_timeout = float(silence_timeout_raw)
    except (TypeError, ValueError):
        silence_timeout = 12.0
    try:
        max_prompts = int(max_prompts_raw)
    except (TypeError, ValueError):
        max_prompts = 3
    silence_timeout = max(5.0, min(60.0, silence_timeout))
    max_prompts = max(1, min(10, max_prompts))

    silence_tracker = {
        "last_user_speech": time.time(),
        "last_agent_speech": time.time(),
        "prompt_count": 0,
        "max_prompts": max_prompts,
        "silence_timeout": silence_timeout,
        "is_waiting_for_response": False,
        "agent_is_speaking": False,
        "enabled": True,
        "paused_by_tool": False,
        "tool_pause_depth": 0,
        "snooze_until": 0.0,
        "last_lookup_progress_prompt_at": 0.0,
        "lookup_progress_prompt_count": 0,
    }
    _current_session["silence_tracker"] = silence_tracker
    
    def reset_silence_timer():
        """Reset the silence timer when user speaks."""
        silence_tracker["last_user_speech"] = time.time()
        silence_tracker["prompt_count"] = 0
        silence_tracker["is_waiting_for_response"] = False
    
    def mark_agent_speaking():
        """Mark that agent just spoke - start silence countdown from now.
        
        We reset BOTH timestamps so the countdown starts fresh after the agent
        finishes speaking. This fixes the bug where the silence prompt would
        trigger immediately after the greeting because last_user_speech was
        set during initialization (before greeting).
        """
        now = time.time()
        silence_tracker["last_agent_speech"] = now
        silence_tracker["last_user_speech"] = now  # Reset user timer too - countdown starts fresh
        silence_tracker["is_waiting_for_response"] = True
    
    import re
    import json as json_module

    async def extract_key_info_ai(text: str) -> list:
        """Keep extraction lightweight for realtime UX."""
        return extract_key_info_fallback(text)

    def extract_key_info_fallback(text: str) -> list:
        """Fallback regex extraction if AI fails."""
        info_items = []
        seen_values = set()
        
        def add_item(item_type, icon, label, value):
            if value and value not in seen_values:
                seen_values.add(value)
                info_items.append({"type": item_type, "icon": icon, "label": label, "value": value})
        
        # Order numbers
        order_matches = re.findall(r'#(\d{4,6})|order\s*(?:number\s*)?#?(\d{4,6})', text, re.I)
        for match in order_matches:
            order_num = match[0] or match[1]
            if order_num:
                add_item("order", "📦", "Order", f"#{order_num}")
        
        # Status
        status_match = re.search(r'\b(delivered|pending|processing|being prepared|shipped|confirmed|cancelled)\b', text, re.I)
        if status_match:
            add_item("status", "✓", "Status", status_match.group(1).title())
        
        # Dates
        date_match = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:,?\s+(\d{4}))?\b', text, re.I)
        if date_match:
            date_str = f"{date_match.group(1)} {date_match.group(2)}"
            if date_match.group(3):
                date_str += f", {date_match.group(3)}"
            add_item("date", "📅", "Date", date_str)
        
        # Prices
        price_match = re.search(r'(\d+(?:[.,]\d{2})?)\s?(?:EUR|€|euros?)', text, re.I)
        if price_match:
            add_item("price", "💰", "Total", f"€{price_match.group(1)}")
        
        # Customer name
        name_match = re.search(r'customer\s+name[^:]*?is\s+([^\.,]+)', text, re.I)
        if name_match:
            add_item("customer", "👤", "Customer", name_match.group(1).strip())
        
        # Ticket
        ticket_match = re.search(r'(TICKET|TKT|TASK)-([A-Z0-9]+)', text, re.I)
        if ticket_match:
            add_item("ticket", "🎫", "Ticket", f"{ticket_match.group(1).upper()}-{ticket_match.group(2)}")
        
        return info_items
    
    async def send_agent_transcript(text: str):
        """Send spoken agent text to frontend chat in realtime."""
        try:
            if call_ended["value"] or _current_session.get("should_end"):
                return
            cleaned = _strip_markup_for_output(text)
            if not cleaned:
                return
            if cleaned in _sent_agent_transcripts:
                return
            _sent_agent_transcripts.add(cleaned)
            if len(_sent_agent_transcripts) > 100:
                _sent_agent_transcripts.clear()
                _sent_agent_transcripts.add(cleaned)

            import json
            transcript_data = json.dumps({
                "type": "transcript",
                "speaker": "agent",
                "text": cleaned,
            })
            await ctx.room.local_participant.publish_data(
                transcript_data.encode('utf-8'),
                reliable=True,
            )
        except Exception as e:
            logger.error(f"Failed to send agent transcript: {e}")

    async def send_agent_info(text: str):
        """Extract and send important information cards without blocking audio."""
        try:
            if call_ended["value"] or _current_session.get("should_end"):
                return
            cleaned = (text or "").strip()
            if not cleaned:
                return

            key_info = await extract_key_info_ai(cleaned)
            if not key_info:
                return

            payload_key = json_module.dumps(key_info, sort_keys=True, ensure_ascii=False)
            if payload_key in _sent_agent_info_payloads:
                return
            _sent_agent_info_payloads.add(payload_key)
            if len(_sent_agent_info_payloads) > 100:
                _sent_agent_info_payloads.clear()
                _sent_agent_info_payloads.add(payload_key)

            import json
            info_data = json.dumps({
                "type": "info",
                "items": key_info,
            })
            await ctx.room.local_participant.publish_data(
                info_data.encode('utf-8'),
                reliable=True,
            )
        except Exception as e:
            logger.error(f"Failed to send agent info: {e}")

    last_lookup_status = {
        "status": None,
        "order_number": None,
        "language": None,
        "updated_at": 0.0,
    }

    def _status_sentence(status: str, language: str) -> str:
        sentences = {
            "en": {
                "processing": "Your order is being prepared.",
                "in_transit": "Your order is on the way.",
                "delivered": "Your order has been delivered.",
                "completed": "Your order is completed.",
                "cancelled": "Your order was cancelled.",
            },
            "el": {
                "processing": "Η παραγγελία σας ετοιμάζεται.",
                "in_transit": "Η παραγγελία σας είναι καθ οδόν.",
                "delivered": "Η παραγγελία σας παραδόθηκε.",
                "completed": "Η παραγγελία σας ολοκληρώθηκε.",
                "cancelled": "Η παραγγελία σας ακυρώθηκε.",
            },
        }
        return sentences.get(language, sentences["en"]).get(status, "")

    def _status_keywords(language: str) -> dict:
        if language == "el":
            return {
                "processing": ["ετοιμάζεται"],
                "in_transit": ["καθ οδόν", "σε μεταφορά", "στο δρόμο"],
                "delivered": ["παραδόθηκε"],
                "completed": ["ολοκληρώθηκε"],
                "cancelled": ["ακυρώθηκε", "ακυρωθηκε"],
            }
        return {
            "processing": ["processing", "being processed", "being prepared", "preparing"],
            "in_transit": ["on the way", "in transit", "out for delivery"],
            "delivered": ["delivered"],
            "completed": ["completed"],
            "cancelled": ["cancelled", "canceled"],
        }

    def _sentence_mentions_status(sentence: str, language: str) -> bool:
        lower = sentence.lower()
        for keywords in _status_keywords(language).values():
            for keyword in keywords:
                if keyword in lower:
                    return True
        return False

    def _enforce_order_status(text: str) -> str:
        if not text:
            return text
        if not last_lookup_status["status"]:
            return text
        # Only enforce shortly after lookup to avoid stale overrides
        if time.time() - last_lookup_status["updated_at"] > 180:
            return text

        language = last_lookup_status["language"] or get_agent_language()
        expected_status = last_lookup_status["status"]
        keywords = _status_keywords(language)
        lower = text.lower()

        found_status = None
        for status, words in keywords.items():
            if any(word in lower for word in words):
                found_status = status
                break

        if not found_status:
            return text
        if found_status == expected_status:
            return text

        status_sentence = _status_sentence(expected_status, language)
        if not status_sentence:
            return text

        # Replace any sentence that mentions a status with the correct one
        parts = re.split(r'(?<=[.!?])\s+', text)
        if len(parts) <= 1:
            return status_sentence

        new_parts = []
        replaced = False
        for sentence in parts:
            if _sentence_mentions_status(sentence, language):
                if not replaced:
                    new_parts.append(status_sentence)
                    replaced = True
                continue
            new_parts.append(sentence)

        if not replaced:
            return status_sentence
        return " ".join(p for p in new_parts if p).strip()

    def _is_details_retry_fallback_text(text: str) -> bool:
        """Detect generic retry/failure filler that appears after details were already spoken."""
        normalized = _normalize_switch_text(text)
        if not normalized:
            return False
        return bool(
            re.search(
                r"(issue retrieving the details|trouble getting the detailed information|let me try again|let me try that again|πρόβλημα .*λεπτομέρει|δυσκολ.*λεπτομέρει|ας το ξαναδοκιμάσω)",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    def _should_suppress_tts_text(candidate_text: str) -> tuple[bool, str]:
        """Shared suppression guard for both string and streaming TTS paths."""
        text_value = (candidate_text or "").strip()
        if not text_value:
            return False, ""

        # Never suppress stock silence prompts with this guard.
        if _is_silence_prompt_text(text_value):
            return False, ""

        if not _current_session.get("forced_response_manual_say_active"):
            suppress_until = float(_current_session.get("forced_response_suppress_llm_until") or 0.0)
            suppress_turn = int(_current_session.get("forced_response_spoken_turn_id") or 0)
            latest_turn = int(_current_session.get("last_user_turn_id") or 0)
            if suppress_until and time.time() <= suppress_until and suppress_turn == latest_turn:
                return True, "forced_response_mutual_exclusion"

        retry_guard_s = _as_float(
            get_agent_setting("forced_details_retry_guard_seconds", 120.0),
            120.0,
            min_value=20.0,
            max_value=300.0,
        )
        last_details_spoken_at = float(_current_session.get("details_last_spoken_at") or 0.0)
        if (
            last_details_spoken_at
            and (time.time() - last_details_spoken_at) <= retry_guard_s
            and _is_details_retry_fallback_text(text_value)
        ):
            return True, "forced_details_retry_guard"

        return False, ""

    def before_tts_callback(agent_instance, text: str | llm.LLMStream):
        """Callback that fires BEFORE text is sent to TTS - apply prosody."""
        try:
            if call_ended["value"] or _current_session.get("should_end"):
                room_log("LATE_EVENT_DROPPED", source="before_tts_callback")
                return ""
            # text can be a string or LLMStream - handle both
            if isinstance(text, str):
                logger.info(f"before_tts_cb processing: {text[:50]}...")
                suppress, suppress_reason = _should_suppress_tts_text(text)
                if suppress:
                    room_log(
                        "TTS_SUPPRESSED",
                        reason=suppress_reason,
                        turn_id=int(_current_session.get("last_user_turn_id") or 0),
                        text=_truncate(text),
                    )
                    return ""
                if _is_lookup_wait_ack_only_text(text):
                    lookup_recent_s = _as_float(
                        get_agent_setting("lookup_wait_ack_require_tool_window_seconds", 4.0),
                        4.0,
                        min_value=1.0,
                        max_value=20.0,
                    )
                    last_lookup_called_at = float(_current_session.get("last_lookup_tool_called_at") or 0.0)
                    lookup_started = bool(
                        _current_session.get("lookup_pending")
                        or _current_session.get("phone_lookup_inflight")
                        or _current_session.get("details_lookup_inflight")
                    ) or (
                        last_lookup_called_at and (time.time() - last_lookup_called_at) <= lookup_recent_s
                    )
                    if not lookup_started:
                        mode = str(_current_session.get("number_mode_lock") or "phone")
                        replacement = _repeat_number_prompt_for_mode(mode, get_agent_language())
                        _current_session["pending_lookup_wait_phrase"] = None
                        _current_session["pending_lookup_wait_phrase_set_at"] = 0.0
                        invalid_recovery_grace = _as_float(
                            get_agent_setting("invalid_number_recovery_silence_grace_seconds", 12.0),
                            12.0,
                            min_value=4.0,
                            max_value=40.0,
                        )
                        _snooze_silence_prompts(invalid_recovery_grace, reason="invalid_number_recovery")
                        room_log(
                            "LOOKUP_WAIT_ACK_REPLACED",
                            reason="lookup_not_started",
                            original=_truncate(text),
                            replacement=_truncate(replacement),
                        )
                        text = replacement
                strict_wait_phrase = _as_bool(
                    get_agent_setting("order_lookup_wait_phrase_strict", True),
                    default=True,
                )
                if strict_wait_phrase:
                    pending_phrase = _current_session.get("pending_lookup_wait_phrase")
                    pending_set_at = float(_current_session.get("pending_lookup_wait_phrase_set_at") or 0.0)
                    wait_phrase_window_s = _as_float(
                        get_agent_setting("lookup_wait_ack_require_tool_window_seconds", 4.0),
                        4.0,
                        min_value=1.0,
                        max_value=20.0,
                    )
                    last_lookup_called_at = float(_current_session.get("last_lookup_tool_called_at") or 0.0)
                    lookup_active = bool(
                        _current_session.get("lookup_pending")
                        or _current_session.get("phone_lookup_inflight")
                        or _current_session.get("details_lookup_inflight")
                    ) or (
                        last_lookup_called_at and (time.time() - last_lookup_called_at) <= wait_phrase_window_s
                    )
                    if pending_phrase and (time.time() - pending_set_at) <= 20.0:
                        if _is_silence_prompt_text(text):
                            room_log("TOOL_WAIT_ACK_SKIPPED", reason="silence_prompt")
                        elif not lookup_active:
                            _clear_pending_lookup_wait_phrase("lookup_not_active")
                        else:
                            if pending_phrase.lower() not in text.lower():
                                text = f"{pending_phrase} {text}".strip()
                                room_log("TOOL_WAIT_ACK_ENFORCED", phrase=_truncate(pending_phrase))
                            _clear_pending_lookup_wait_phrase("consumed")
                    elif pending_phrase:
                        # Expire stale pending phrase to avoid polluting unrelated turns.
                        _clear_pending_lookup_wait_phrase("expired")
                text = _enforce_locked_output_language(text)

                from src.utils import (
                    apply_prosody,
                    normalize_time_colons,
                    normalize_numeric_ids_for_tts,
                    normalize_punctuation_for_tts,
                )
                agent_lang = get_agent_language()
                use_ssml = _as_bool(get_agent_setting("tts_use_ssml", False), default=False)
                tts_engine = getattr(agent_instance, "_tts", None) or getattr(agent_instance, "tts", None)
                tts_provider = "unknown"
                supports_ssml = False
                if tts_engine is not None:
                    if hasattr(tts_engine, "current_provider_name"):
                        try:
                            tts_provider = tts_engine.current_provider_name()
                        except Exception:
                            tts_provider = str(getattr(tts_engine, "_provider_name", "unknown"))
                    else:
                        tts_provider = str(getattr(tts_engine, "_provider_name", "unknown"))

                    if hasattr(tts_engine, "supports_ssml"):
                        try:
                            supports_ssml = bool(tts_engine.supports_ssml())
                        except Exception:
                            supports_ssml = bool(getattr(tts_engine, "_supports_ssml", False))
                    else:
                        supports_ssml = bool(getattr(tts_engine, "_supports_ssml", False))
                if tts_provider == "unknown":
                    tts_provider = str(_current_session.get("tts_provider") or "unknown")

                if use_ssml and not supports_ssml:
                    room_log(
                        "SSML_DISABLED_FOR_PROVIDER",
                        requested=True,
                        provider=tts_provider,
                    )
                    use_ssml = False

                tts_text = _enforce_order_status(text)
                sanitized_tts_text = _strip_tts_style_leakage(tts_text)
                if sanitized_tts_text != tts_text:
                    room_log(
                        "TTS_STYLE_TEXT_STRIPPED",
                        before=_truncate(tts_text, max_len=200),
                        after=_truncate(sanitized_tts_text, max_len=200),
                    )
                tts_text = sanitized_tts_text
                tts_text = normalize_time_colons(tts_text)
                tts_text = normalize_numeric_ids_for_tts(tts_text, language=agent_lang)
                tts_text = normalize_punctuation_for_tts(tts_text)
                cleaned_ui_text = _strip_markup_for_output(tts_text)
                if cleaned_ui_text:
                    room_log(
                        "TTS_TEXT_FINAL",
                        provider=tts_provider,
                        ssml=use_ssml,
                        text=_truncate(cleaned_ui_text),
                    )
                processed_text = apply_prosody(tts_text, language=agent_lang, use_ssml=use_ssml)
                return processed_text
            else:
                logger.debug("before_tts_cb got LLMStream (applying streaming normalization)")
                from src.utils import (
                    normalize_time_colons,
                    normalize_numeric_ids_for_tts,
                    normalize_punctuation_for_tts,
                )
                agent_lang = get_agent_language()

                def _extract_chunk_text(chunk) -> str:
                    if isinstance(chunk, str):
                        return chunk
                    for attr in ("text", "delta", "content"):
                        if hasattr(chunk, attr):
                            value = getattr(chunk, attr)
                            if isinstance(value, str):
                                return value
                    return str(chunk)

                async def _normalized_stream(stream):
                    raw_buffer = ""
                    normalized_buffer = ""
                    async for chunk in stream:
                        chunk_text = _extract_chunk_text(chunk)
                        if not chunk_text:
                            continue
                        raw_buffer += chunk_text
                        updated = _enforce_order_status(raw_buffer)
                        updated = _strip_tts_style_leakage(updated)
                        updated = _enforce_locked_output_language(updated)
                        updated = normalize_time_colons(updated)
                        updated = normalize_numeric_ids_for_tts(updated, language=agent_lang)
                        updated = normalize_punctuation_for_tts(updated)
                        suppress, suppress_reason = _should_suppress_tts_text(updated)
                        if suppress:
                            room_log(
                                "TTS_SUPPRESSED",
                                reason=suppress_reason,
                                turn_id=int(_current_session.get("last_user_turn_id") or 0),
                                text=_truncate(updated),
                            )
                            return
                        if updated.startswith(normalized_buffer):
                            delta = updated[len(normalized_buffer):]
                        else:
                            delta = updated
                        normalized_buffer = updated
                        if delta:
                            yield delta

                return _normalized_stream(text)
        except Exception as e:
            logger.error(f"Error in before_tts_callback: {e}")
        return text  # Return text unchanged on error
    # Wait for context that was started earlier (should be ready by now)
    initial_ctx = await context_task
    logger.info(f"⏱️ Context ready ({time.time() - startup_time:.1f}s)")
    
    # Create the voice pipeline agent - tuned to avoid clipping user speech.
    min_endpointing_delay = _as_float(
        get_agent_setting("min_endpointing_delay", 0.40),
        0.40,
        min_value=0.2,
        max_value=1.5,
    )
    interrupt_min_words = _as_int(
        get_agent_setting("interrupt_min_words", 3),
        3,
        min_value=1,
        max_value=10,
    )
    logger.info(
        "Turn config: min_endpointing_delay=%.2fs interrupt_min_words=%s",
        min_endpointing_delay,
        interrupt_min_words,
    )
    logger.info(
        "Language switch config: auto_language_switch=%s min_turns=%s call_type=%s",
        auto_language_switch,
        language_switch_min_turns,
        call_type,
    )
    preemptive_synthesis = _as_bool(
        get_agent_setting("preemptive_synthesis", True),
        default=True,
    )
    logger.info("TTS preemptive_synthesis=%s", preemptive_synthesis)
    logger.info(f"⏱️ Creating agent ({time.time() - startup_time:.1f}s)")
    agent = VoicePipelineAgent(
        vad=create_vad(),
        stt=create_stt(is_sip_call=is_sip_call),
        llm=create_llm(),
        tts=create_tts(),
        chat_ctx=initial_ctx,
        fnc_ctx=ElenaFunctionContext(),
        max_nested_fnc_calls=3,
        # Turn segmentation settings.
        min_endpointing_delay=min_endpointing_delay,
        preemptive_synthesis=preemptive_synthesis,  # Allow toggling to avoid punctuation leaks
        allow_interruptions=True,       # Allow user to interrupt
        interrupt_min_words=interrupt_min_words,
        before_tts_cb=before_tts_callback,  # Capture text when possible
    )
    logger.info(f"⏱️ Agent created ({time.time() - startup_time:.1f}s)")
    
    # ==========================================================================
    # LATENCY TRACKING - Register event handlers
    # ==========================================================================
    async def send_state_update(state: str):
        """Helper to send agent state updates to frontend."""
        try:
            import json
            state_data = json.dumps({
                "type": "state",
                "state": state
            })
            await ctx.room.local_participant.publish_data(
                state_data.encode('utf-8'),
                reliable=True
            )
            logger.debug(f"🎯 Agent state sent: {state}")
        except Exception as e:
            logger.error(f"Failed to send state update: {e}")
    
    thinking_task: Optional[asyncio.Task] = None

    def cancel_thinking_task():
        nonlocal thinking_task
        if thinking_task and not thinking_task.done():
            thinking_task.cancel()
        thinking_task = None

    def schedule_thinking_state(delay_s: float = 0.35):
        nonlocal thinking_task
        cancel_thinking_task()

        async def _set_thinking():
            try:
                await asyncio.sleep(delay_s)
                await send_state_update("thinking")
            except asyncio.CancelledError:
                return

        thinking_task = asyncio.create_task(_set_thinking())

    @agent.on("user_started_speaking")
    def on_user_started_speaking():
        _latency_tracker.user_started_speaking()
        cancel_thinking_task()
        asyncio.create_task(send_state_update("listening"))
        # Immediately stop silence detection when user starts talking
        # This prevents "are you there?" from interrupting the user mid-speech
        silence_tracker["is_waiting_for_response"] = False
        silence_tracker["last_user_speech"] = time.time()
    
    @agent.on("user_stopped_speaking")
    def on_user_stopped_speaking():
        _latency_tracker.user_stopped_speaking()
        schedule_thinking_state()

    async def send_user_transcript(text: str, *, interim: bool = False):
        """Helper to send user transcript to frontend."""
        try:
            if call_ended["value"] or _current_session.get("should_end"):
                return
            cleaned = (text or "").strip()
            if not cleaned:
                return

            nonlocal _last_user_interim, _last_user_interim_sent_at, _last_user_final
            now = time.monotonic()

            if interim:
                # Throttle interim updates to avoid flooding the UI.
                if cleaned == _last_user_interim and (now - _last_user_interim_sent_at) < 0.35:
                    return
                _last_user_interim = cleaned
                _last_user_interim_sent_at = now
            else:
                # Avoid duplicate finals, but always override interim if present.
                if cleaned == _last_user_final and cleaned != _last_user_interim:
                    return
                _last_user_final = cleaned
                _last_user_interim = ""

            import json
            transcript_data = json.dumps({
                "type": "transcript",
                "speaker": "user",
                "text": cleaned,
                "interim": interim,
            })
            await ctx.room.local_participant.publish_data(
                transcript_data.encode('utf-8'),
                reliable=True
            )
            logger.debug(f"?? User transcript sent: {cleaned[:50]}...")
            if not interim:
                room_log("USER_TEXT", text=_truncate(cleaned))
        except Exception as e:
            logger.error(f"Failed to send user transcript: {e}")

    async def _force_get_order_details(turn_id: int, trigger_reason: str) -> None:
        """Deterministically fetch and speak order details after explicit unlock."""
        if call_ended["value"] or _current_session.get("should_end"):
            room_log("ORDER_DETAILS_FORCED_SKIP", reason="call_ended", turn_id=turn_id)
            return

        if bool(_current_session.get("details_lookup_inflight")):
            room_log("ORDER_DETAILS_FORCED_SKIP", reason="already_inflight", turn_id=turn_id)
            return

        last_forced_turn = int(_current_session.get("details_forced_turn_id") or 0)
        if turn_id <= last_forced_turn:
            room_log(
                "ORDER_DETAILS_FORCED_SKIP",
                reason="duplicate_turn",
                turn_id=turn_id,
                last_forced_turn=last_forced_turn,
            )
            return

        expected = _expected_order_digits()
        order_number = re.sub(r"\D", "", str(_current_session.get("last_lookup_order") or ""))
        if not re.fullmatch(rf"\d{{{expected}}}", order_number):
            room_log(
                "ORDER_DETAILS_FORCED_SKIP",
                reason="missing_anchor_order",
                order_number=order_number,
                turn_id=turn_id,
            )
            return

        last_spoken_order = re.sub(r"\D", "", str(_current_session.get("details_last_spoken_order") or ""))
        last_spoken_at = float(_current_session.get("details_last_spoken_at") or 0.0)
        if (
            order_number
            and last_spoken_order == order_number
            and (time.time() - last_spoken_at) <= 20.0
        ):
            room_log(
                "ORDER_DETAILS_FORCED_SKIP",
                reason="recently_spoken",
                order_number=order_number,
                turn_id=turn_id,
            )
            return

        forced_suppress_s = _as_float(
            get_agent_setting("forced_details_llm_suppress_seconds", 120.0),
            120.0,
            min_value=20.0,
            max_value=300.0,
        )
        _current_session["details_forced_pending_turn_id"] = turn_id
        _current_session["details_lookup_inflight"] = True
        _current_session["details_forced_turn_id"] = turn_id
        _current_session["forced_response_spoken_turn_id"] = turn_id
        _current_session["forced_response_suppress_llm_until"] = time.time() + forced_suppress_s
        _set_lookup_pending(order_number, reason="forced_get_order_details")
        _pause_silence_for_tool("forced_get_order_details")

        try:
            room_log(
                "TOOL_CALL",
                name="get_order_details",
                order_number=order_number,
                forced=True,
                trigger=trigger_reason,
            )
            result = await order_lookup.get_order_details(order_number)
            room_log(
                "TOOL_RESULT",
                name="get_order_details",
                result=_truncate(result),
                forced=True,
                trigger=trigger_reason,
            )

            spoken_summary = _build_order_details_voice_summary(result, get_agent_language())
            if not spoken_summary:
                spoken_summary = _build_order_voice_summary(result, get_agent_language()) or (
                    "I could not find details for this order."
                    if get_agent_language() == "en"
                    else "Δεν μπόρεσα να βρω λεπτομέρειες για αυτή την παραγγελία."
                )

            prefix = (
                "Thanks for waiting."
                if get_agent_language() == "en"
                else "Ευχαριστώ για την αναμονή."
            )
            final_text = f"{prefix} {spoken_summary}".strip()
            room_log(
                "ORDER_DETAILS_FORCED_FORMATTED",
                order_number=order_number,
                result=_truncate(final_text),
            )
            room_log("ORDER_DETAILS_FORCED_SPEAKING", order_number=order_number, turn_id=turn_id)
            _current_session["forced_response_manual_say_active"] = True
            _current_session["forced_response_spoken_text"] = final_text
            await agent.say(final_text, allow_interruptions=True)
            _current_session["forced_response_suppress_llm_until"] = time.time() + forced_suppress_s
            mark_agent_speaking()
            _snooze_silence_prompts(10.0, reason="post_forced_order_details_spoken")
            _clear_lookup_pending(reason="forced_order_details_spoken")
            _current_session["details_last_spoken_order"] = order_number
            _current_session["details_last_spoken_at"] = time.time()
            room_log("ORDER_DETAILS_FORCED_SPOKEN", order_number=order_number, turn_id=turn_id)
        except Exception as e:
            room_log(
                "ORDER_DETAILS_FORCED_ERROR",
                order_number=order_number,
                trigger=trigger_reason,
                error=_truncate(str(e), max_len=200),
            )
            _clear_lookup_pending(reason="forced_order_details_error")
        finally:
            _current_session["forced_response_manual_say_active"] = False
            _current_session["details_lookup_inflight"] = False
            if int(_current_session.get("details_forced_pending_turn_id") or 0) == turn_id:
                _current_session["details_forced_pending_turn_id"] = 0
            _resume_silence_for_tool("forced_get_order_details")

    async def _force_lookup_by_phone(turn_id: int, phone: str, trigger_reason: str) -> None:
        """Deterministically run lookup_order_by_phone and speak the result."""
        if call_ended["value"] or _current_session.get("should_end"):
            room_log("PHONE_LOOKUP_FORCED_SKIP", reason="call_ended", turn_id=turn_id)
            return

        flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
        if flow_state != FLOW_CHECKING_PHONE_NUMBER:
            room_log(
                "PHONE_LOOKUP_FORCED_BLOCKED",
                reason="phone_not_confirmed",
                flow_state=flow_state,
                phone=phone,
            )
            _clear_lookup_pending("phone_not_confirmed")
            _current_session["phone_lookup_inflight"] = False
            return

        normalized_phone = _normalize_phone_for_lookup(phone or "")
        if not normalized_phone:
            room_log(
                "PHONE_LOOKUP_FORCED_SKIP",
                reason="invalid_phone_pattern",
                turn_id=turn_id,
                phone=_truncate(phone, max_len=64),
            )
            return

        if bool(_current_session.get("phone_lookup_inflight")):
            room_log("PHONE_LOOKUP_FORCED_SKIP", reason="already_inflight", turn_id=turn_id)
            return

        last_forced_turn = int(_current_session.get("phone_forced_turn_id") or 0)
        if turn_id <= last_forced_turn:
            room_log(
                "PHONE_LOOKUP_FORCED_SKIP",
                reason="duplicate_turn",
                turn_id=turn_id,
                last_forced_turn=last_forced_turn,
            )
            return

        forced_suppress_s = _as_float(
            get_agent_setting("forced_phone_llm_suppress_seconds", 90.0),
            90.0,
            min_value=15.0,
            max_value=300.0,
        )
        _current_session["phone_forced_pending_turn_id"] = turn_id
        _current_session["phone_lookup_inflight"] = True
        _current_session["phone_forced_turn_id"] = turn_id
        _current_session["forced_response_spoken_turn_id"] = turn_id
        _current_session["forced_response_suppress_llm_until"] = time.time() + forced_suppress_s
        _set_support_flow_state(FLOW_CHECKING_PHONE_NUMBER, reason=f"forced_phone_lookup:{trigger_reason}")
        _set_lookup_pending(normalized_phone, reason="phone_lookup_started")
        _snooze_silence_prompts(45.0, reason="phone_lookup_started")
        _current_session["lookup_progress_prompt_until"] = time.time() + 45.0
        _current_session["last_lookup_tool_called_at"] = time.time()
        _current_session["last_lookup_tool_order"] = normalized_phone
        _pause_silence_for_tool("forced_lookup_order_by_phone")
        
        wait_msg = "Μισό λεπτό, ψάχνω την παραγγελία σας." if get_agent_language() == "el" else "Just a moment, I am searching for your order."
        await agent.say(wait_msg, allow_interruptions=False)

        try:
            room_log(
                "TOOL_CALL",
                name="lookup_order_by_phone",
                phone=normalized_phone,
                forced=True,
                trigger=trigger_reason,
            )
            result = await order_lookup.lookup_order_by_phone(normalized_phone)
            room_log(
                "TOOL_RESULT",
                name="lookup_order_by_phone",
                result=_truncate(result),
                forced=True,
                trigger=trigger_reason,
            )

            lookup_state = _classify_lookup_result(result)
            _current_session["last_lookup_state"] = lookup_state
            snapshot = order_lookup.get_last_order_snapshot() or {}
            snapshot_order = str(snapshot.get("order_number") or "")
            strict_snapshot_order = _normalize_order_id_strict(snapshot_order) if snapshot_order else None
            _current_session["last_lookup_order"] = (
                strict_snapshot_order if (lookup_state == "found" and strict_snapshot_order) else ""
            )
            if lookup_state == "found":
                _set_support_flow_state(FLOW_ORDER_FOUND, reason=f"forced_phone_lookup:{trigger_reason}")
                _current_session["details_confirmation_pending"] = True
                _current_session["details_confirmation_pending_until"] = time.time() + 120.0
            else:
                _set_support_flow_state(FLOW_AWAITING_PHONE_NUMBER, reason=f"forced_phone_lookup_{lookup_state}")
                _current_session["details_confirmation_pending"] = False
                _current_session["details_confirmation_pending_until"] = 0.0
                _current_session["full_order_details_allowed_until"] = 0.0

            spoken_summary = _build_phone_lookup_voice_summary(result, get_agent_language()) or result
            room_log(
                "PHONE_LOOKUP_FORCED_FORMATTED",
                phone=normalized_phone,
                result=_truncate(spoken_summary),
            )
            room_log("PHONE_LOOKUP_FORCED_SPEAKING", phone=normalized_phone, turn_id=turn_id)
            _current_session["forced_response_manual_say_active"] = True
            _current_session["forced_response_spoken_text"] = spoken_summary
            await agent.say(spoken_summary, allow_interruptions=True)
            _current_session["forced_response_suppress_llm_until"] = time.time() + forced_suppress_s
            mark_agent_speaking()
            _snooze_silence_prompts(10.0, reason="post_forced_phone_lookup_spoken")
            room_log("PHONE_LOOKUP_FORCED_SPOKEN", phone=normalized_phone, turn_id=turn_id)
        except Exception as e:
            room_log(
                "PHONE_LOOKUP_FORCED_ERROR",
                phone=normalized_phone,
                trigger=trigger_reason,
                error=_truncate(str(e), max_len=200),
            )
        finally:
            _current_session["forced_response_manual_say_active"] = False
            _current_session["phone_lookup_inflight"] = False
            _clear_lookup_pending(reason="phone_lookup_finished")
            _clear_pending_lookup_wait_phrase("phone_lookup_finished")
            _reset_phone_digit_buffer("phone_lookup_finished")
            _snooze_silence_prompts(5.0, reason="phone_lookup_finished")
            if int(_current_session.get("phone_forced_pending_turn_id") or 0) == turn_id:
                _current_session["phone_forced_pending_turn_id"] = 0
            _resume_silence_for_tool("forced_lookup_order_by_phone")

    async def _speak_phone_confirmation_prompt(
        turn_id: int,
        phone: str,
        trigger_reason: str,
        *,
        reprompt: bool = False,
    ) -> None:
        """Speak phone confirmation directly from code, independent of LLM instruction following."""
        try:
            if call_ended["value"] or _current_session.get("should_end"):
                room_log("PHONE_CONFIRMATION_SKIP", reason="call_ended", turn_id=turn_id)
                return

            normalized_phone = _normalize_phone_for_lookup(phone or "")
            if not normalized_phone:
                room_log(
                    "PHONE_CONFIRMATION_SKIP",
                    reason="invalid_phone_pattern",
                    turn_id=turn_id,
                    phone=_truncate(phone, max_len=64),
                )
                return

            latest_turn = int(_current_session.get("last_user_turn_id") or 0)
            if latest_turn > turn_id:
                room_log(
                    "PHONE_CONFIRMATION_SKIP",
                    reason="newer_user_turn",
                    turn_id=turn_id,
                    latest_turn=latest_turn,
                )
                return

            pending_phone = str(_current_session.get("pending_phone_candidate") or "")
            if pending_phone and pending_phone != normalized_phone:
                room_log(
                    "PHONE_CONFIRMATION_SKIP",
                    reason="pending_phone_mismatch",
                    turn_id=turn_id,
                    pending_phone=pending_phone,
                    phone=normalized_phone,
                )
                return

            if not _is_phone_confirmation_pending():
                room_log("PHONE_CONFIRMATION_SKIP", reason="not_awaiting_confirmation", turn_id=turn_id)
                return

            suppress_s = _as_float(
                get_agent_setting("phone_confirmation_llm_suppress_seconds", 18.0),
                18.0,
                min_value=8.0,
                max_value=60.0,
            )
            confirmation_snooze_s = _as_float(
                get_agent_setting("phone_confirmation_silence_snooze_seconds", 10.0),
                10.0,
                min_value=4.0,
                max_value=30.0,
            )

            _current_session["forced_response_spoken_turn_id"] = turn_id
            _current_session["forced_response_suppress_llm_until"] = time.time() + suppress_s
            spoken_phone = _speak_digits(normalized_phone, get_agent_language())

            if get_agent_language() == "el":
                if reprompt:
                    confirmation_text = (
                        f"Για να συνεχίσουμε, απαντήστε μόνο ναι ή όχι. Ο αριθμός είναι {spoken_phone}. Είναι σωστός;"
                    )
                else:
                    confirmation_text = f"Για επιβεβαίωση, ο αριθμός τηλεφώνου σας είναι {spoken_phone}. Είναι σωστός;"
            else:
                if reprompt:
                    confirmation_text = (
                        f"To continue, please answer only yes or no. The number is {spoken_phone}. Is that correct?"
                    )
                else:
                    confirmation_text = f"Just to confirm, your phone number is {spoken_phone}. Is that correct?"

            room_log(
                "PHONE_CONFIRMATION_PROMPT",
                turn_id=turn_id,
                phone=normalized_phone,
                reason=trigger_reason,
                reprompt=reprompt,
            )

            _current_session["forced_response_manual_say_active"] = True
            _current_session["forced_response_spoken_text"] = confirmation_text
            await agent.say(confirmation_text, allow_interruptions=True)
            _current_session["forced_response_suppress_llm_until"] = time.time() + suppress_s
            mark_agent_speaking()
            _snooze_silence_prompts(confirmation_snooze_s, reason="phone_confirmation_prompt")
            room_log("PHONE_CONFIRMATION_SPOKEN", turn_id=turn_id, reason=trigger_reason)
        finally:
            _current_session["forced_response_manual_say_active"] = False

    @agent.on("user_speech_committed")
    def on_user_speech_committed(message):
        """Send user transcript to frontend and check for abuse."""
        if call_ended["value"] or _current_session.get("should_end"):
            room_log("LATE_EVENT_DROPPED", source="user_speech_committed")
            return
        user_text = message.content
        current_turn_id = int(_current_session.get("last_user_turn_id") or 0) + 1
        _current_session["last_user_turn_id"] = current_turn_id
        flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
        phone_flow_active = flow_state in PHONE_FLOW_STATES
        is_affirmative = _is_affirmative_utterance(user_text)
        is_negative = _is_negative_utterance(user_text)
        normalized_order_candidate = (
            _normalize_order_id_strict(user_text)
            if flow_state == FLOW_AWAITING_ORDER_NUMBER
            else None
        )

        phone_lookup_schedule_snooze = _as_float(
            get_agent_setting("phone_lookup_schedule_snooze_seconds", 20.0),
            20.0,
            min_value=5.0,
            max_value=90.0,
        )

        # Clear stale pending forced-turn markers from previous turns.
        if (
            int(_current_session.get("phone_forced_pending_turn_id") or 0) < current_turn_id
            and not bool(_current_session.get("phone_lookup_inflight"))
        ):
            _current_session["phone_forced_pending_turn_id"] = 0
        if (
            int(_current_session.get("details_forced_pending_turn_id") or 0) < current_turn_id
            and not bool(_current_session.get("details_lookup_inflight"))
        ):
            _current_session["details_forced_pending_turn_id"] = 0
        # New user turn cancels stale forced-response suppression window.
        last_forced_response_turn = int(_current_session.get("forced_response_spoken_turn_id") or 0)
        if current_turn_id > last_forced_response_turn:
            _current_session["forced_response_suppress_llm_until"] = 0.0
            _current_session["forced_response_spoken_text"] = ""

        def _handle_phone_flow_turn(
            *,
            user_text: str,
            current_turn_id: int,
            is_affirmative: bool,
            is_negative: bool,
        ) -> bool:
            """Deterministic phone-flow state machine. Returns True when turn is fully handled."""
            local_flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
            if local_flow_state not in PHONE_FLOW_STATES:
                return False

            raw_digits = "".join(_extract_digit_parts(user_text or ""))
            pending_phone = str(_current_session.get("pending_phone_candidate") or "")

            def _schedule_manual_prompt(
                message_text: str,
                *,
                reason: str,
                suppress_s: float = 8.0,
                snooze_s: float = 8.0,
            ) -> None:
                _current_session["forced_response_manual_say_active"] = True
                _current_session["forced_response_spoken_turn_id"] = current_turn_id
                _current_session["forced_response_suppress_llm_until"] = time.time() + suppress_s
                _current_session["forced_response_spoken_text"] = message_text
                _clear_pending_lookup_wait_phrase(reason)
                _clear_lookup_pending(reason)
                _current_session["phone_lookup_inflight"] = False
                _snooze_silence_prompts(snooze_s, reason=reason)

                async def _say_prompt() -> None:
                    try:
                        await agent.say(message_text, allow_interruptions=True)
                        _current_session["forced_response_suppress_llm_until"] = time.time() + suppress_s
                        mark_agent_speaking()
                    finally:
                        _current_session["forced_response_manual_say_active"] = False

                asyncio.create_task(_say_prompt())

            def _schedule_phone_confirmation_from_turn(phone_candidate: str, trigger_reason: str) -> bool:
                """Only confirm phone when current turn actually contains digits."""
                if not raw_digits:
                    room_log(
                        "PHONE_CONFIRMATION_BLOCKED",
                        reason="no_digits_in_current_user_turn",
                        user_text=_truncate(user_text),
                        stale_pending_phone=_current_session.get("pending_phone_candidate"),
                        stale_buffer=_current_session.get("phone_digit_buffer"),
                    )
                    _reset_phone_collection_state("blocked_stale_phone_confirmation")
                    _set_support_flow_state(
                        FLOW_AWAITING_PHONE_NUMBER,
                        reason="blocked_stale_phone_confirmation",
                    )
                    return False

                _reset_phone_digit_buffer("phone_candidate_captured")
                _current_session["pending_phone_candidate"] = phone_candidate
                _set_support_flow_state(FLOW_AWAITING_PHONE_CONFIRMATION, reason="phone_candidate_captured")
                room_log("PHONE_CANDIDATE_CAPTURED", phone=phone_candidate, turn_id=current_turn_id)
                _current_session["forced_response_manual_say_active"] = True
                _current_session["forced_response_spoken_turn_id"] = current_turn_id
                _current_session["forced_response_suppress_llm_until"] = time.time() + 8.0
                _clear_pending_lookup_wait_phrase("phone_confirmation_prompt")
                _clear_lookup_pending("phone_confirmation_prompt")
                _current_session["phone_lookup_inflight"] = False
                _snooze_silence_prompts(8.0, reason="phone_confirmation_prompt")
                asyncio.create_task(
                    _speak_phone_confirmation_prompt(
                        current_turn_id,
                        phone_candidate,
                        trigger_reason,
                    )
                )
                return True

            if local_flow_state == FLOW_CHECKING_PHONE_NUMBER:
                room_log("PHONE_FLOW_TURN_IGNORED", reason="lookup_already_running", turn_id=current_turn_id)
                return True

            if local_flow_state == FLOW_AWAITING_PHONE_NUMBER:
                if not raw_digits:
                    if get_agent_language() == "el":
                        msg = "Παρακαλώ πείτε τον αριθμό τηλεφώνου που χρησιμοποιήσατε για την παραγγελία, ψηφίο προς ψηφίο."
                    else:
                        msg = "Please provide the phone number used for the order, digit by digit."
                    _schedule_manual_prompt(msg, reason="awaiting_phone_digits")
                    return True

                combined_digits = _append_phone_digits_from_turn(user_text)
                phone_candidate = _normalize_phone_for_lookup(combined_digits)
                if phone_candidate:
                    return _schedule_phone_confirmation_from_turn(
                        phone_candidate,
                        "phone_candidate_captured",
                    )

                min_digits = _as_int(
                    get_agent_setting("phone_lookup_min_digits", 10),
                    10,
                    min_value=7,
                    max_value=15,
                )
                if len(combined_digits) < min_digits:
                    if get_agent_language() == "el":
                        msg = "Σας ακούω. Συνεχίστε με τα υπόλοιπα ψηφία του τηλεφώνου, παρακαλώ."
                    else:
                        msg = "I’m listening. Please continue with the remaining digits of the phone number."
                    _schedule_manual_prompt(msg, reason="phone_digits_partial", suppress_s=6.0)
                    return True

                _reset_phone_digit_buffer("invalid_complete_phone")
                _current_session["pending_phone_candidate"] = None
                _set_support_flow_state(FLOW_AWAITING_PHONE_NUMBER, reason="invalid_complete_phone")
                if get_agent_language() == "el":
                    msg = (
                        "Αυτό δεν φαίνεται να είναι πλήρης αριθμός τηλεφώνου. "
                        f"Παρακαλώ επαναλάβετε ολόκληρο τον αριθμό, τουλάχιστον {min_digits} ψηφία, ψηφίο προς ψηφίο."
                    )
                else:
                    msg = (
                        "That does not look like a complete phone number. "
                        f"Please repeat the full number, at least {min_digits} digits, digit by digit."
                    )
                _schedule_manual_prompt(msg, reason="invalid_complete_phone")
                room_log("INVALID_OR_PARTIAL_PHONE_REJECTED", digits=combined_digits, turn_id=current_turn_id)
                return True

            # FLOW_AWAITING_PHONE_CONFIRMATION
            phone_candidate = _normalize_phone_for_lookup(user_text or "")
            if phone_candidate:
                return _schedule_phone_confirmation_from_turn(
                    phone_candidate,
                    "phone_candidate_captured",
                )

            if pending_phone and is_affirmative:
                _current_session["pending_phone_candidate"] = None
                _set_support_flow_state(FLOW_CHECKING_PHONE_NUMBER, reason="phone_confirmed")
                _current_session["forced_response_spoken_turn_id"] = current_turn_id
                _current_session["forced_response_suppress_llm_until"] = time.time() + 30.0
                _current_session["phone_forced_pending_turn_id"] = current_turn_id
                _set_lookup_pending(pending_phone, reason="forced_phone_lookup_scheduled")
                _snooze_silence_prompts(phone_lookup_schedule_snooze, reason="phone_lookup_scheduled")
                asyncio.create_task(
                    _force_lookup_by_phone(current_turn_id, pending_phone, "phone_confirmed")
                )
                room_log("PHONE_LOOKUP_FORCED_TRIGGERED", phone=pending_phone, turn_id=current_turn_id)
                return True

            if is_negative:
                _current_session["pending_phone_candidate"] = None
                _reset_phone_digit_buffer("phone_confirmation_rejected")
                _set_support_flow_state(FLOW_AWAITING_PHONE_NUMBER, reason="phone_confirmation_rejected")
                if get_agent_language() == "el":
                    msg = "Εντάξει. Πείτε ξανά τον αριθμό τηλεφώνου σας, ψηφίο προς ψηφίο."
                else:
                    msg = "Okay. Please repeat your phone number again, digit by digit."
                _schedule_manual_prompt(msg, reason="phone_confirmation_rejected")
                return True

            if raw_digits:
                combined_digits = _append_phone_digits_from_turn(user_text)
                min_digits = _as_int(
                    get_agent_setting("phone_lookup_min_digits", 10),
                    10,
                    min_value=7,
                    max_value=15,
                )
                if len(combined_digits) < min_digits:
                    if get_agent_language() == "el":
                        msg = "Σας ακούω. Συνεχίστε με τα υπόλοιπα ψηφία του τηλεφώνου, παρακαλώ."
                    else:
                        msg = "I’m listening. Please continue with the remaining digits of the phone number."
                    _schedule_manual_prompt(msg, reason="phone_digits_partial", suppress_s=6.0)
                else:
                    _reset_phone_digit_buffer("invalid_complete_phone")
                    if get_agent_language() == "el":
                        msg = (
                            "Αυτό δεν φαίνεται να είναι πλήρης αριθμός τηλεφώνου. "
                            f"Παρακαλώ επαναλάβετε ολόκληρο τον αριθμό, τουλάχιστον {min_digits} ψηφία, ψηφίο προς ψηφίο."
                        )
                    else:
                        msg = (
                            "That does not look like a complete phone number. "
                            f"Please repeat the full number, at least {min_digits} digits, digit by digit."
                        )
                    _schedule_manual_prompt(msg, reason="invalid_complete_phone")
                return True

            if pending_phone:
                if not raw_digits:
                    room_log(
                        "PHONE_CONFIRMATION_BLOCKED",
                        reason="no_digits_in_current_user_turn",
                        user_text=_truncate(user_text),
                        stale_pending_phone=_current_session.get("pending_phone_candidate"),
                        stale_buffer=_current_session.get("phone_digit_buffer"),
                    )
                    _reset_phone_collection_state("blocked_stale_phone_confirmation")
                    _set_support_flow_state(
                        FLOW_AWAITING_PHONE_NUMBER,
                        reason="blocked_stale_phone_confirmation",
                    )
                    return True
                return _schedule_phone_confirmation_from_turn(
                    pending_phone,
                    "non_confirmation_reply",
                )
            if get_agent_language() == "el":
                msg = "Παρακαλώ πείτε ξανά τον αριθμό τηλεφώνου σας, ψηφίο προς ψηφίο."
            else:
                msg = "Please say your phone number again, digit by digit."
            _schedule_manual_prompt(msg, reason="awaiting_phone_recovery")
            return True

        if phone_flow_active:
            handled = _handle_phone_flow_turn(
                user_text=user_text,
                current_turn_id=current_turn_id,
                is_affirmative=is_affirmative,
                is_negative=is_negative,
            )
            if handled:
                return

        # Deterministic state bootstrap: treat the first open-ended complaint as issue context,
        # then ask for order number unless caller explicitly says they don't have it.
        if flow_state == FLOW_IDLE and user_text and not _is_short_utterance(user_text):
            if not _mentions_no_order_number(user_text) and not normalized_order_candidate:
                _current_session["support_issue"] = user_text
                _set_support_flow_state(FLOW_AWAITING_ORDER_NUMBER, reason="issue_collected")
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "SUPPORT FLOW:\n"
                        "- The issue has been collected.\n"
                        "- Ask the caller for their order number next.\n"
                        "- If they don't have an order number, ask for the phone number used in the order."
                    ),
                )

        has_digits = bool(_extract_digit_parts(user_text))
        raw_digits = "".join(_extract_digit_parts(user_text or ""))
        expected_digits = _expected_order_digits()
        if flow_state == FLOW_AWAITING_ORDER_NUMBER and raw_digits:
            normalized_order = _normalize_order_id_strict(user_text or "")
            if not normalized_order:
                _current_session["number_mode_lock"] = "order"
                _set_support_flow_state(
                    FLOW_AWAITING_ORDER_NUMBER,
                    reason="invalid_order_digits",
                )
                _current_session["forced_response_manual_say_active"] = True
                _current_session["forced_response_spoken_turn_id"] = current_turn_id
                _current_session["forced_response_suppress_llm_until"] = time.time() + 8.0
                _clear_lookup_pending("invalid_order_digits")
                _clear_pending_lookup_wait_phrase("invalid_order_digits")
                _snooze_silence_prompts(8.0, reason="invalid_order_digits")

                async def _say_invalid_order_digits() -> None:
                    live_agent = _current_session.get("agent")
                    if not live_agent:
                        _current_session["forced_response_manual_say_active"] = False
                        return
                    try:
                        if get_agent_language() == "el":
                            msg = (
                                f"Ο αριθμός παραγγελίας πρέπει να έχει ακριβώς "
                                f"{expected_digits} ψηφία. Μπορείτε να τον επαναλάβετε "
                                f"ψηφίο προς ψηφίο;"
                            )
                        else:
                            msg = (
                                f"The order number must be exactly {expected_digits} digits. "
                                f"Could you repeat it digit by digit?"
                            )
                        await live_agent.say(msg, allow_interruptions=True)
                    finally:
                        _current_session["forced_response_manual_say_active"] = False

                asyncio.create_task(_say_invalid_order_digits())
                return

        if (
            flow_state == FLOW_AWAITING_ORDER_NUMBER
            and not _mentions_no_order_number(user_text)
            and not normalized_order_candidate
        ):
            expected = _expected_order_digits()
            if has_digits:
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "SUPPORT FLOW - ORDER NUMBER STILL MISSING:\n"
                        f"- The caller has not provided a valid {expected}-digit order number yet.\n"
                        "- Ask them to repeat the order number digit by digit.\n"
                        "- Do not switch to phone lookup unless they explicitly say they don't have an order number."
                    ),
                )
            else:
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "SUPPORT FLOW - REQUEST ORDER NUMBER:\n"
                        "- Ask for the order number now.\n"
                        "- If caller says they don't have it, then request the phone number used in the order."
                    ),
                )

        if _mentions_no_order_number(user_text):
            _reset_phone_collection_state("user_has_no_order_number")
            _clear_lookup_pending("user_has_no_order_number")
            _clear_pending_lookup_wait_phrase("user_has_no_order_number")
            _current_session["number_mode_lock"] = "phone"
            _set_support_flow_state(
                FLOW_AWAITING_PHONE_NUMBER,
                reason="user_has_no_order_number",
            )
            _current_session["forced_response_manual_say_active"] = True
            _current_session["forced_response_spoken_turn_id"] = current_turn_id
            _current_session["forced_response_suppress_llm_until"] = time.time() + 8.0
            _snooze_silence_prompts(8.0, reason="ask_phone_after_no_order_number")

            async def _ask_phone_number() -> None:
                live_agent = _current_session.get("agent")
                if not live_agent:
                    _current_session["forced_response_manual_say_active"] = False
                    return
                try:
                    if get_agent_language() == "el":
                        msg = (
                            "Κατανοητό. Μπορείτε να μου δώσετε τον αριθμό τηλεφώνου "
                            "που χρησιμοποιήσατε για την παραγγελία, ψηφίο προς ψηφίο;"
                        )
                    else:
                        msg = (
                            "No problem. Please give me the phone number used for the order, "
                            "digit by digit."
                        )
                    await live_agent.say(msg, allow_interruptions=True)
                finally:
                    _current_session["forced_response_manual_say_active"] = False

            asyncio.create_task(_ask_phone_number())
            return
        elif normalized_order_candidate and flow_state == FLOW_AWAITING_ORDER_NUMBER:
            # Caller provided a valid order id, so clear phone-capture context.
            _current_session["pending_phone_candidate"] = None
            _reset_phone_digit_buffer("back_to_order_flow")
            _set_support_flow_state(FLOW_CHECKING_ORDER_NUMBER, reason="order_number_provided")
        _current_session["number_mode_lock"] = None
        _current_session["number_mode_turn_id"] = 0
        flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
        inferred_mode = _infer_number_mode(user_text, str(_current_session.get("last_agent_text") or ""))
        phone_flow_states = PHONE_FLOW_STATES
        if (
            flow_state == FLOW_AWAITING_ORDER_NUMBER
            and inferred_mode == "phone"
            and not _mentions_no_order_number(user_text)
        ):
            inferred_mode = None
            room_log("NUMBER_MODE_SUPPRESSED", flow_state=flow_state, candidate="phone")
        if (
            inferred_mode is None
            and _is_phone_flow_active()
            and bool(_extract_digit_parts(user_text))
        ):
            inferred_mode = "phone"
        if (
            flow_state in phone_flow_states
            and inferred_mode == "order"
            and not normalized_order_candidate
        ):
            inferred_mode = "phone"
            room_log("NUMBER_MODE_FORCED", flow_state=flow_state, mode="phone")
        if inferred_mode in {"order", "phone"}:
            _current_session["number_mode_lock"] = inferred_mode
            _current_session["number_mode_turn_id"] = current_turn_id
            room_log("NUMBER_MODE_LOCKED", mode=inferred_mode, turn_id=current_turn_id)

        now = time.time()
        explicit_details_request = _explicit_more_order_details_request(user_text)

        # Drop stale phone candidate if this turn is clearly back to order-id flow.
        if inferred_mode == "order":
            _current_session["pending_phone_candidate"] = None

        details_pending = bool(_current_session.get("details_confirmation_pending"))
        details_pending_until = float(_current_session.get("details_confirmation_pending_until") or 0.0)
        if details_pending and now <= details_pending_until:
            if explicit_details_request or is_affirmative:
                can_unlock, unlock_reason = _can_unlock_full_details(user_text)
                if can_unlock:
                    _current_session["full_order_details_allowed_until"] = now + 120.0
                    _current_session["full_details_unlocked_once"] = True
                    _current_session["details_confirmation_pending"] = False
                    _current_session["details_confirmation_pending_until"] = 0.0
                    room_log("FULL_DETAILS_ALLOWED", ttl_s=120, reason="single_yes_unlock")
                    # Deterministic path: suppress generic LLM chatter and force details tool fetch.
                    _current_session["forced_response_spoken_turn_id"] = current_turn_id
                    _current_session["forced_response_suppress_llm_until"] = time.time() + 20.0
                    _current_session["details_forced_pending_turn_id"] = current_turn_id
                    asyncio.create_task(
                        _force_get_order_details(current_turn_id, "single_yes_unlock")
                    )
                else:
                    room_log("FULL_DETAILS_BLOCKED", reason=unlock_reason)
                    _current_session["full_order_details_allowed_until"] = 0.0
                    _current_session["details_confirmation_pending"] = False
                    _current_session["details_confirmation_pending_until"] = 0.0
            elif is_negative:
                _current_session["details_confirmation_pending"] = False
                _current_session["details_confirmation_pending_until"] = 0.0
                _current_session["full_order_details_allowed_until"] = 0.0
        elif explicit_details_request:
            can_unlock, unlock_reason = _can_unlock_full_details(user_text)
            if can_unlock:
                _current_session["full_order_details_allowed_until"] = now + 120.0
                _current_session["full_details_unlocked_once"] = True
                _current_session["details_confirmation_pending"] = False
                _current_session["details_confirmation_pending_until"] = 0.0
                room_log("FULL_DETAILS_ALLOWED", ttl_s=120, reason="explicit_request")
                _current_session["forced_response_spoken_turn_id"] = current_turn_id
                _current_session["forced_response_suppress_llm_until"] = time.time() + 20.0
                _current_session["details_forced_pending_turn_id"] = current_turn_id
                asyncio.create_task(
                    _force_get_order_details(current_turn_id, "explicit_request")
                )
            else:
                room_log("FULL_DETAILS_BLOCKED", reason=unlock_reason)

        ticket_pending = bool(_current_session.get("ticket_confirmation_pending"))
        ticket_pending_until = float(_current_session.get("ticket_confirmation_pending_until") or 0.0)
        if ticket_pending and now <= ticket_pending_until:
            if _is_issue_confirmation_utterance(user_text):
                _current_session["ticket_create_allowed_until"] = now + 120.0
                _current_session["ticket_confirmation_pending"] = False
                _current_session["ticket_confirmation_pending_until"] = 0.0
                room_log("TICKET_CREATE_ALLOWED", ttl_s=120, reason="issue_confirmed")
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "SUPPORT TICKET CONFIRMED:\n"
                        "- The user gave final confirmation for the issue.\n"
                        "- Call create_support_ticket now with the collected customer details.\n"
                        "- Create the ticket only once."
                    ),
                )
            elif is_negative:
                _current_session["ticket_confirmation_pending"] = False
                _current_session["ticket_confirmation_pending_until"] = 0.0
                _current_session["ticket_create_allowed_until"] = 0.0
                _current_session["pending_ticket_payload"] = None

        if auto_language_switch:
            explicit_lang = _explicit_language_request(user_text)
            if explicit_lang and explicit_lang != session_language["value"]:
                room_log(
                    "LANGUAGE_SWITCH_INTENT",
                    current=session_language["value"],
                    requested=explicit_lang,
                    reason="explicit_request",
                )
                _apply_language_switch(explicit_lang, reason="explicit_request")
                # Extra hard override so this turn switches immediately.
                lang_name = "English" if explicit_lang == "en" else "Greek"
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "HIGHEST PRIORITY LANGUAGE OVERRIDE:\n"
                        f"- The user explicitly requested {lang_name}.\n"
                        f"- Reply in {lang_name} immediately for this response.\n"
                        "- Do not refuse. Do not say you only speak another language."
                    ),
                )
            else:
                detected_lang = detect_language(user_text, default=session_language["value"])
                if detected_lang != session_language["value"] and not _allow_auto_language_switch(
                    session_language["value"],
                    detected_lang,
                    user_text,
                ):
                    detected_lang = session_language["value"]
                room_log(
                    "USER_TEXT_LANG",
                    current=session_language["value"],
                    detected=detected_lang,
                    explicit=explicit_lang or "",
                )
                if detected_lang != session_language["value"]:
                    if language_switch_state["candidate"] == detected_lang:
                        language_switch_state["count"] += 1
                    else:
                        language_switch_state["candidate"] = detected_lang
                        language_switch_state["count"] = 1

                    room_log(
                        "LANGUAGE_SWITCH_CANDIDATE",
                        current=session_language["value"],
                        candidate=detected_lang,
                        count=language_switch_state["count"],
                        required=language_switch_min_turns,
                    )

                    if language_switch_state["count"] >= language_switch_min_turns:
                        _apply_language_switch(detected_lang, reason="consecutive_detected")
                else:
                    language_switch_state["candidate"] = None
                    language_switch_state["count"] = 0
                    set_runtime_language(detected_lang)
        else:
            explicit_lang = _explicit_language_request(user_text)
            if explicit_lang and explicit_lang != session_language["value"]:
                room_log(
                    "LANGUAGE_SWITCH_SUPPRESSED",
                    current=session_language["value"],
                    candidate=explicit_lang,
                    reason="auto_switch_disabled",
                )
                locked_name = "Greek" if session_language["value"] == "el" else "English"
                agent.chat_ctx.append(
                    role="system",
                    text=(
                        "LANGUAGE LOCK:\n"
                        "- Auto language switching is disabled for this call.\n"
                        f"- Reply only in {locked_name}.\n"
                        f"- If the caller requests another language, politely continue in {locked_name}."
                    ),
                )

        try:
            detected_order_number = _extract_order_number_candidate(user_text)
            number_mode = str(_current_session.get("number_mode_lock") or "")
            flow_state = str(_current_session.get("support_flow_state") or FLOW_IDLE)
            order_lookup_blocked_by_flow = (
                flow_state in {FLOW_AWAITING_PHONE_CONFIRMATION, FLOW_CHECKING_PHONE_NUMBER}
                and not detected_order_number
            )
            if (
                _should_force_order_lookup(user_text, detected_order_number)
                and number_mode != "phone"
                and not order_lookup_blocked_by_flow
            ):
                now = time.time()
                last_forced_order = str(_current_session.get("last_forced_lookup_order") or "")
                last_forced_at = float(_current_session.get("last_forced_lookup_at") or 0.0)
                if (
                    detected_order_number != last_forced_order
                    or (now - last_forced_at) > 15.0
                ):
                    _current_session["last_forced_lookup_order"] = detected_order_number
                    _current_session["last_forced_lookup_at"] = now
                    agent.chat_ctx.append(
                        role="system",
                        text=(
                            "ORDER LOOKUP PRIORITY:\n"
                            f"- The caller already provided order number {detected_order_number}.\n"
                            "- In your next response, call lookup_order with this exact number immediately.\n"
                            "- Do not ask for the order number again unless lookup_order says invalid or not found.\n"
                            "- After the tool returns, provide the status without extra delay."
                        ),
                    )
                    room_log("ORDER_LOOKUP_HINT_INJECTED", order_number=detected_order_number)

                    # No background forced-response task here to avoid duplicate/racing
                    # responses with normal tool calls in the same turn.
            elif order_lookup_blocked_by_flow and _should_force_order_lookup(user_text, detected_order_number):
                room_log(
                    "ORDER_LOOKUP_HINT_SKIPPED",
                    reason="phone_flow_in_progress",
                    flow_state=flow_state,
                )
        except Exception as e:
            logger.debug(f"Order lookup forcing skipped: {e}")

        user_text_for_transcript = _format_user_text_for_transcript(user_text)
        asyncio.create_task(send_user_transcript(user_text_for_transcript))

        # Reset silence timer - user is responding
        reset_silence_timer()
        if _is_digit_collection_utterance(user_text):
            digit_grace = _as_float(
                get_agent_setting("digit_collection_silence_grace_seconds", 14.0),
                14.0,
                min_value=4.0,
                max_value=40.0,
            )
            _snooze_silence_prompts(digit_grace, reason="digit_collection")
        elif _is_short_utterance(user_text):
            short_grace = _as_float(
                get_agent_setting("short_utterance_silence_grace_seconds", 6.0),
                6.0,
                min_value=2.0,
                max_value=20.0,
            )
            _snooze_silence_prompts(short_grace, reason="short_utterance")

        # Add to transcript
        conversation_transcript.append(f"User: {user_text_for_transcript}")
        if abuse_detection_enabled:
            # Check for abusive language
            abuse_detected, abuse_response = check_and_respond_to_abuse(
                user_text,
                language=get_agent_language(),
                tracker=_abuse_tracker,
                use_ssml=True
            )

            if abuse_detected:
                logger.warning("Abuse detected in: %s...", user_text[:50])
                # The agent will continue normally, but we log the incident
                # The abuse response will be handled by the LLM with special instructions
                # For now, we just track it for escalation purposes

    @agent.on("agent_started_speaking")
    def on_agent_started_speaking():
        _latency_tracker.agent_started_speaking()
        cancel_thinking_task()
        logger.info("audio_publish_start: agent_started_speaking")
        silence_tracker["agent_is_speaking"] = True
        # Never fire silence prompts while the agent is actively speaking.
        silence_tracker["is_waiting_for_response"] = False
        asyncio.create_task(send_state_update("speaking"))
    
    @agent.on("agent_stopped_speaking")
    def on_agent_stopped_speaking():
        silence_tracker["agent_is_speaking"] = False
        asyncio.create_task(send_state_update("idle"))
        # Mark that agent finished speaking - now waiting for user response
        mark_agent_speaking()
    
    @agent.on("agent_speech_committed")
    def on_agent_speech_committed(message):
        """Send committed agent speech to UI (transcript + info cards)."""
        try:
            if call_ended["value"] or _current_session.get("should_end"):
                room_log("LATE_EVENT_DROPPED", source="agent_speech_committed")
                return
            text = message.content if hasattr(message, 'content') else None
            if text:
                normalized_text = _enforce_locked_output_language(text)
                display_text = _strip_markup_for_output(normalized_text)
                suppress_commit, suppress_reason = _should_suppress_tts_text(display_text or text)
                if suppress_commit:
                    # Keep the deterministic forced-response summary in transcript even inside suppression windows.
                    expected_forced_response_text = str(_current_session.get("forced_response_spoken_text") or "")
                    expected_norm = _normalize_switch_text(expected_forced_response_text)
                    candidate_norm = _normalize_switch_text(display_text or text)
                    same_as_forced_response = bool(
                        expected_norm
                        and candidate_norm
                        and (
                            candidate_norm == expected_norm
                            or candidate_norm in expected_norm
                            or expected_norm in candidate_norm
                        )
                    )
                    if not same_as_forced_response:
                        room_log(
                            "AGENT_TEXT_SUPPRESSED",
                            reason=suppress_reason,
                            turn_id=int(_current_session.get("last_user_turn_id") or 0),
                            text=_truncate(display_text or text),
                        )
                        return
                transcript_text = _format_agent_text_for_transcript(display_text or text)
                asyncio.create_task(send_agent_transcript(transcript_text))
                asyncio.create_task(send_agent_info(transcript_text))
                conversation_transcript.append(f"Agent: {transcript_text}")
                _current_session["last_agent_text"] = transcript_text
                normalized_display = _normalize_switch_text(transcript_text)
                details_prompted = bool(
                    re.search(
                        r"(would you like .*details.*order|complete order details|more details about this order|θέλετε .*λεπτομέρειες.*παραγγελία)",
                        normalized_display,
                    )
                )
                if details_prompted:
                    _current_session["last_more_details_prompt_at"] = time.time()
                    if not bool(_current_session.get("full_details_unlocked_once")):
                        _current_session["details_confirmation_pending"] = True
                        _current_session["details_confirmation_pending_until"] = time.time() + 120.0
                    else:
                        _current_session["details_confirmation_pending"] = False
                        _current_session["details_confirmation_pending_until"] = 0.0
                if _is_clarification_prompt_text(transcript_text):
                    clarification_grace = _as_float(
                        get_agent_setting("clarification_silence_grace_seconds", 12.0),
                        12.0,
                        min_value=4.0,
                        max_value=40.0,
                    )
                    _snooze_silence_prompts(clarification_grace, reason="clarification_prompt")
                if _current_session.get("lookup_pending"):
                    lookup_state_in_text = _classify_lookup_result(transcript_text)
                    if details_prompted or lookup_state_in_text in {"found", "not_found"}:
                        _clear_lookup_pending(reason="agent_committed_lookup_result")
                logger.info(f"agent_speech_committed: {transcript_text[:50]}...")
                room_log("AGENT_TEXT", text=_truncate(transcript_text))
        except Exception as e:
            logger.error(f"Error in agent_speech_committed: {e}")
    # Track detailed metrics from pipeline
    @agent.on("metrics_collected")
    def on_metrics_collected(metrics):
        """Collect detailed metrics from the pipeline - logs individual service timings."""
        try:
            parts = []
            
            # Try different attribute names (SDK versions vary)
            # STT metrics
            stt_duration = getattr(metrics, 'stt_duration', None) or getattr(metrics, 'transcription_delay', None)
            if stt_duration:
                parts.append(f"STT:{stt_duration*1000:.0f}ms")
            
            # LLM metrics
            llm_ttft = getattr(metrics, 'llm_ttft', None) or getattr(metrics, 'llm_first_token_delay', None)
            if llm_ttft:
                parts.append(f"LLM-TTFT:{llm_ttft*1000:.0f}ms")
            
            llm_total = getattr(metrics, 'llm_duration', None) or getattr(metrics, 'llm_total_duration', None)
            if llm_total:
                parts.append(f"LLM-Total:{llm_total*1000:.0f}ms")
            
            # TTS metrics
            tts_ttfb = getattr(metrics, 'tts_ttfb', None) or getattr(metrics, 'speech_start_delay', None)
            if tts_ttfb:
                parts.append(f"TTS:{tts_ttfb*1000:.0f}ms")
            
            # End-to-end delay
            eou = getattr(metrics, 'eou_delay', None) or getattr(metrics, 'end_of_utterance_delay', None)
            if eou:
                parts.append(f"EoU:{eou*1000:.0f}ms")
            
            if parts:
                logger.info(f"📊 [METRICS] {' | '.join(parts)}")
                room_log("METRICS", details=" | ".join(parts))
            else:
                # Log all available metrics for debugging
                attrs = [a for a in dir(metrics) if not a.startswith('_')]
                logger.debug(f"Available metrics: {attrs}")
                
        except Exception as e:
            logger.debug(f"Metrics collection error: {e}")
    
    # Additional event for function calls timing
    @agent.on("function_calls_finished")
    def on_function_calls_finished(called_functions):
        """Track tool call execution time."""
        for fn in called_functions:
            # Try different attribute names for the function name
            name = (
                getattr(fn, 'function_name', None) or 
                getattr(fn, 'name', None) or 
                getattr(fn, 'tool_name', None) or
                getattr(fn, '__name__', None) or
                str(type(fn).__name__)
            )
            result = getattr(fn, 'result', None)
            logger.info(f"🔧 [TOOL] {name} executed" + (f" - result: {str(result)[:100]}" if result else ""))
            room_log("TOOL_EXECUTED", name=name, result=_truncate(str(result)) if result else None)
            if name == "lookup_order":
                snapshot = order_lookup.get_last_order_snapshot()
                if snapshot:
                    last_lookup_status["status"] = snapshot.get("status")
                    last_lookup_status["order_number"] = snapshot.get("order_number")
                    last_lookup_status["language"] = get_agent_language()
                    last_lookup_status["updated_at"] = time.time()
    
    # Store references for session management
    _current_session["agent"] = agent
    _current_session["room"] = ctx.room
    
    async def handle_call_end(reason: str = "normal"):
        """Handle call ending - save transcript and clean up."""
        if call_ended["value"]:
            logger.debug("Call end already handled, skipping")
            return
        call_ended["value"] = True
        _current_session["should_end"] = True
        
        try:
            # Calculate call duration
            call_duration = int(time.time() - call_start_time)
            
            # Build transcript string
            full_transcript = "\n".join(conversation_transcript)
            
            logger.info(f"Handling call end: {reason}, duration={call_duration}s, transcript_lines={len(conversation_transcript)}")
            room_log("CALL_END", reason=reason, duration_s=call_duration, transcript_lines=len(conversation_transcript))
            room_log("FULL_TRANSCRIPT", transcript=full_transcript)
            
            # Log call completed event (for all calls)
            await log_call_event(
                event_type="call_completed",
                room_name=ctx.room.name,
                call_type=call_type,
                caller_number=caller_number,
                caller_identity=caller_identity,
                call_id=db_call_id,
                duration_seconds=call_duration,
                metadata={
                    "source": "livekit_agent",
                    "disconnect_reason": reason,
                    "transcript_lines": len(conversation_transcript),
                },
            )
            
            # Record call end in database (updates analytics) with transcript
            await end_call_in_db(
                call_id=db_call_id,
                room_name=ctx.room.name,
                status="completed",
                duration_seconds=call_duration,
                disconnect_reason=reason,
                transcript=full_transcript if full_transcript else None,
            )
            
            logger.info(f"Call recorded: {db_call_id}, transcript saved ({len(full_transcript)} chars)")
            
        except Exception as e:
            logger.error(f"Error handling call end: {e}")
        finally:
            _current_session["silence_tracker"] = None
            _current_session["silence_pause_depth"] = 0
            _current_session["last_silence_block_reason"] = None
            _current_session["last_lookup_wait_phrase"] = None
            _current_session["pending_lookup_wait_phrase"] = None
            _current_session["pending_lookup_wait_phrase_set_at"] = 0.0
            _current_session["last_agent_text"] = ""
            _current_session["last_forced_lookup_order"] = None
            _current_session["last_forced_lookup_at"] = 0.0
            _current_session["last_user_turn_id"] = 0
            _current_session["last_lookup_tool_called_at"] = 0.0
            _current_session["last_lookup_tool_order"] = None
            _current_session["lookup_progress_prompt_until"] = 0.0
            _current_session["number_mode_lock"] = None
            _current_session["number_mode_turn_id"] = 0
            _current_session["forced_response_manual_say_active"] = False
            _current_session["forced_response_spoken_turn_id"] = 0
            _current_session["forced_response_spoken_text"] = ""
            _current_session["forced_response_suppress_llm_until"] = 0.0
            _current_session["lookup_pending"] = False
            _current_session["lookup_pending_started_at"] = 0.0
            _current_session["lookup_pending_order"] = None
            _current_session["last_lookup_state"] = "unknown"
            _current_session["last_lookup_order"] = None
            _current_session["pending_phone_candidate"] = None
            _current_session["phone_lookup_inflight"] = False
            _current_session["phone_forced_turn_id"] = 0
            _current_session["phone_forced_pending_turn_id"] = 0
            _current_session["details_lookup_inflight"] = False
            _current_session["details_forced_turn_id"] = 0
            _current_session["details_forced_pending_turn_id"] = 0
            _current_session["details_last_spoken_order"] = None
            _current_session["details_last_spoken_at"] = 0.0
            _current_session["details_confirmation_pending"] = False
            _current_session["details_confirmation_pending_until"] = 0.0
            _current_session["full_order_details_allowed_until"] = 0.0
            _current_session["full_details_unlocked_once"] = False
            _current_session["last_more_details_prompt_at"] = 0.0
            _current_session["ticket_confirmation_pending"] = False
            _current_session["ticket_confirmation_pending_until"] = 0.0
            _current_session["ticket_create_allowed_until"] = 0.0
            _current_session["pending_ticket_payload"] = None
            _current_session["ticket_created"] = False
            _current_session["ticket_reference"] = None
            _current_session["support_flow_state"] = FLOW_IDLE
            _current_session["support_issue"] = None
            set_runtime_language(None)
    # Handle participant disconnection
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant_info):
        """Handle when a participant (user) disconnects."""
        logger.info(f"Participant disconnected: {participant_info.identity}")
        
        # Check if this is the main participant (not the agent)
        if participant_info.identity != "agent" and participant_info.identity == caller_identity:
            logger.info("User disconnected - ending call and closing room")
            asyncio.create_task(handle_call_end("user_disconnected"))
            # Disconnect room after a short delay to allow cleanup
            async def delayed_disconnect():
                await asyncio.sleep(0.5)
                try:
                    if ctx.room and ctx.room.isconnected():
                        await ctx.room.disconnect()
                except Exception as e:
                    logger.debug(f"Room disconnect: {e}")
            asyncio.create_task(delayed_disconnect())
    
    # Handle room disconnection
    @ctx.room.on("disconnected")
    def on_room_disconnected():
        """Handle when the room is disconnected."""
        logger.info("Room disconnected")
        asyncio.create_task(handle_call_end("room_disconnected"))
    
    agent.start(ctx.room, participant)
    logger.info(f"⏱️ Agent started ({time.time() - startup_time:.1f}s)")

    # Stream interim user transcripts to the UI for realtime feel.
    human_input = getattr(agent, "_human_input", None)
    if human_input:
        @human_input.on("interim_transcript")
        def on_interim_transcript(ev):
            try:
                text = ev.alternatives[0].text
            except Exception:
                text = None
            if text:
                cancel_thinking_task()
                asyncio.create_task(send_user_transcript(text, interim=True))
    
    # Start background audio if enabled (runs completely independently)
    bg_audio_player = None
    async def start_background_audio():
        nonlocal bg_audio_player
        try:
            from src.services.background_audio import create_background_audio_player
            bg_audio_player = await create_background_audio_player()
            if bg_audio_player:
                success = await bg_audio_player.start(ctx.room)
                if success:
                    logger.info("🎵 Background audio playing continuously")
                else:
                    logger.warning("🎵 Background audio failed to start")
        except Exception as e:
            logger.debug(f"Background audio not available: {e}")
    
    # Get greeting based on configured language
    agent_lang = get_agent_language()
    greeting_enabled = _require_bool_setting("agent_greeting_enabled")
    if greeting_enabled:
        greeting = get_greeting(agent_lang)
        logger.info(f"⏱️ Saying greeting ({time.time() - startup_time:.1f}s): {greeting[:50]}...")
        await agent.say(greeting, allow_interruptions=True)
    else:
        logger.info("Greeting disabled by settings")
    
    total_startup = time.time() - startup_time
    logger.info(f"✅ Elena ready! Total startup: {total_startup:.1f}s, language: {agent_lang}")
    
    # Reset silence timer after greeting (or initial ready state if greeting disabled)
    mark_agent_speaking()

    # Start background audio after the greeting to avoid delaying first response.
    bg_audio_task = asyncio.create_task(start_background_audio())
    bg_audio_task.set_name("bg_audio_init")

    # Defer order prefetch until after greeting to reduce initial latency.
    asyncio.create_task(order_lookup.prefetch_orders())
    
    # =========================================================================
    # SILENCE MONITORING - Prompt user if no response
    # =========================================================================
    async def monitor_silence():
        """Monitor for user silence and prompt them."""
        agent_lang = get_agent_language()
        
        # Silence prompts based on language
        if agent_lang == "el":
            prompts = [
                "Είμαι εδώ όταν είστε έτοιμοι.",
                "Πάρτε τον χρόνο σας. Είμαι ακόμη εδώ.",
                "Θα τερματίσω την κλήση προς το παρόν. Μπορείτε να μας καλέσετε ξανά οποιαδήποτε στιγμή.",
            ]
        else:
            prompts = [
                "I’m here when you’re ready.",
                "Take your time. I’m still here.",
                "I’ll end the call for now. You can call us again anytime.",
            ]
        try:
            while not _current_session["should_end"] and silence_tracker["enabled"]:
                await asyncio.sleep(1.0)  # Check every second
                
                # Only check silence if we're waiting for a response
                if not silence_tracker["is_waiting_for_response"]:
                    continue

                now = time.time()

                lookup_active = (
                    bool(_current_session.get("lookup_pending"))
                    or bool(_current_session.get("phone_lookup_inflight"))
                    or bool(_current_session.get("details_lookup_inflight"))
                )
                if lookup_active:
                    pending_started = float(_current_session.get("lookup_pending_started_at") or 0.0)
                    max_lookup_block_s = _as_float(
                        get_agent_setting("lookup_silence_block_max_seconds", 60.0),
                        60.0,
                        min_value=15.0,
                        max_value=180.0,
                    )

                    # If lookup state is stale, clear it silently.
                    if pending_started and (time.time() - pending_started) > max_lookup_block_s:
                        room_log(
                            "LOOKUP_SILENCE_BLOCK_STALE_CLEARED",
                            age_s=round(time.time() - pending_started, 2),
                        )
                        _clear_lookup_pending("lookup_silence_block_stale")
                        _current_session["phone_lookup_inflight"] = False
                        _current_session["details_lookup_inflight"] = False
                        tracker = _current_session.get("silence_tracker")
                        if isinstance(tracker, dict):
                            tracker["last_user_speech"] = time.time()
                            tracker["last_agent_speech"] = time.time()
                            tracker["prompt_count"] = 0

                    # Important:
                    # While lookup is active, silence monitor must not speak.
                    continue

                # Pause silence prompts while tool calls are executing.
                if silence_tracker.get("paused_by_tool"):
                    continue

                # Skip checks while agent audio is still being rendered.
                if silence_tracker.get("agent_is_speaking"):
                    continue

                # Strong global guard: never say silence prompts during deterministic work.
                if _should_block_silence_prompt("monitor_loop"):
                    continue
                
                # Calculate time since last activity
                time_since_user = now - silence_tracker["last_user_speech"]
                time_since_agent = now - silence_tracker["last_agent_speech"]
                
                # Only trigger if:
                # 1. User hasn't spoken for silence_timeout seconds
                # 2. Agent finished speaking at least silence_timeout seconds ago
                if time_since_user >= silence_tracker["silence_timeout"] and \
                   time_since_agent >= silence_tracker["silence_timeout"]:
                    
                    prompt_count = silence_tracker["prompt_count"]
                    
                    if prompt_count < silence_tracker["max_prompts"]:
                        # Prompt the user
                        prompt_text = prompts[min(prompt_count, len(prompts) - 1)]
                        # Race-condition guard right before speaking.
                        if _should_block_silence_prompt("before_silence_prompt_say"):
                            continue
                        logger.info(f"🔇 Silence detected ({time_since_user:.1f}s), prompting user: {prompt_text}")
                        
                        silence_tracker["prompt_count"] += 1
                        silence_tracker["is_waiting_for_response"] = False  # Will be set again after agent speaks
                        
                        # Say the prompt
                        await agent.say(prompt_text, allow_interruptions=True)
                        
                    else:
                        # Max prompts reached - disconnect
                        goodbye_text = prompts[-1]  # Last prompt is goodbye
                        logger.info(f"🔇 Max silence prompts reached, disconnecting: {goodbye_text}")
                        
                        silence_tracker["enabled"] = False
                        await agent.say(goodbye_text, allow_interruptions=False)
                        
                        # Wait for goodbye to finish, then disconnect
                        await asyncio.sleep(3.0)
                        _current_session["should_end"] = True
                        break
                        
        except asyncio.CancelledError:
            logger.debug("Silence monitor cancelled")
        except Exception as e:
            logger.error(f"Silence monitor error: {e}")
    
    # Monitor for session end request (agent-initiated end)
    async def monitor_session_end():
        try:
            while not _current_session["should_end"]:
                await asyncio.sleep(0.5)
            
            logger.info("Session end flag set - disconnecting now")
            await asyncio.sleep(0.5)
            
            # Stop background audio
            if bg_audio_player:
                await bg_audio_player.stop()
            
            await handle_call_end("agent_ended")
            
            # Clean disconnect
            if ctx.room and ctx.room.isconnected():
                await ctx.room.disconnect()
            
            logger.info("Session ended by agent successfully")
            
        except Exception as e:
            logger.debug(f"Session end cleanup: {e} (expected during disconnect)")
    
    # Start monitoring tasks in background
    asyncio.create_task(monitor_silence())
    asyncio.create_task(monitor_session_end())


def prewarm(proc: JobProcess):
    """
    Prewarm function - Keep it lightweight to avoid issues.
    
    NOTE: We intentionally do NOT pre-fetch database here because:
    1. It creates connections in a separate event loop that can't be reused
    2. It exhausts the database connection pool
    3. The connections get "attached to a different loop" errors
    
    The database fetch happens quickly during call startup with parallel queries.
    """
    logger.info("Prewarm: ready (lightweight)")

    def _warm_tts():
        # Best-effort TTS warm-up to reduce first greeting latency.
        try:
            import urllib.request
            import urllib.error

            # Warm ElevenLabs if configured.
            if settings.elevenlabs_api_key:
                voice_id = settings.elevenlabs_voice_id
                model_id = settings.elevenlabs_model or "eleven_multilingual_v2"
                payload = {
                    "text": "Hi",
                    "model_id": model_id,
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    data=data,
                    headers={
                        "xi-api-key": settings.elevenlabs_api_key,
                        "Content-Type": "application/json",
                        "accept": "audio/mpeg",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=4) as resp:
                        resp.read(1)
                    logger.info("TTS prewarm: ElevenLabs warmed")
                except Exception as e:
                    logger.debug(f"TTS prewarm: ElevenLabs skipped ({e})")

            # Warm OpenAI TTS if configured.
            if settings.openai_api_key:
                payload = {
                    "model": "tts-1",
                    "voice": "alloy",
                    "input": "Hi",
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.openai.com/v1/audio/speech",
                    data=data,
                    headers={
                        "Authorization": f"Bearer {settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=4) as resp:
                        resp.read(1)
                    logger.info("TTS prewarm: OpenAI warmed")
                except Exception as e:
                    logger.debug(f"TTS prewarm: OpenAI skipped ({e})")
        except Exception as e:
            logger.debug(f"TTS prewarm error: {e}")

    threading.Thread(target=_warm_tts, daemon=True).start()


def run_agent():
    """
    Run the Elena voice agent as a LiveKit worker.
    """
    import os

    def _get_float_env(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _get_int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    initialize_timeout = _get_float_env("LIVEKIT_AGENTS_INITIALIZE_TIMEOUT", 60.0)
    shutdown_timeout = _get_float_env("LIVEKIT_AGENTS_SHUTDOWN_TIMEOUT", 60.0)
    num_idle_processes = _get_int_env("LIVEKIT_AGENTS_NUM_IDLE_PROCESSES", 1)
    load_threshold = _get_float_env("LIVEKIT_AGENTS_LOAD_THRESHOLD", 0.90)

    initialize_timeout = max(5.0, min(300.0, initialize_timeout))
    shutdown_timeout = max(10.0, min(300.0, shutdown_timeout))
    num_idle_processes = max(0, min(8, num_idle_processes))
    load_threshold = max(0.1, min(1.0, load_threshold))

    logger.info(
        "Worker options: initialize_timeout=%ss, shutdown_timeout=%ss, num_idle_processes=%s, load_threshold=%s",
        initialize_timeout,
        shutdown_timeout,
        num_idle_processes,
        load_threshold,
    )

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
            initialize_process_timeout=initialize_timeout,
            shutdown_process_timeout=shutdown_timeout,
            num_idle_processes=num_idle_processes,
            load_threshold=load_threshold,
        ),
    )


# For running directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_agent()








