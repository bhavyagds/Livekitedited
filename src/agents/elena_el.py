"""
Meallion Voice AI - Elena Greek Agent (clean rewrite)
Greek-only voice agent with deterministic order/phone support flow.
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Optional

from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import elevenlabs, openai, silero

try:
    from livekit.plugins import deepgram
    USE_DEEPGRAM = True
except ImportError:
    USE_DEEPGRAM = False

from src.config import settings
from src.agents.prompts import (
    _fetch_from_db,
    get_agent_setting,
    get_closing,
    get_greeting,
    get_system_prompt_async,
    set_runtime_language,
)
from src.agents.tools import knowledge_base, order_lookup, support_ticket

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small utils
# -----------------------------------------------------------------------------

def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_float(value: object, default: float, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _as_int(value: object, default: int, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = default
    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _truncate(text: str, max_len: int = 180) -> str:
    s = str(text or "")
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


_ORDER_WORDS = {
    "μηδέν": "0", "ένα": "1", "δύο": "2", "τρία": "3", "τέσσερα": "4",
    "πέντε": "5", "έξι": "6", "επτά": "7", "οκτώ": "8", "εννέα": "9",
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

_MEMORY_STOPWORDS = {
    "ο", "η", "το", "οι", "τα", "του", "της", "των", "τον", "την", "στο", "στη", "στην",
    "και", "είναι", "για", "με", "από", "σε", "που", "να", "θα", "δεν", "μη", "μου", "σου", "του",
}


def _extract_digit_parts(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    parts: list[str] = []
    for token in tokens:
        if token in _ORDER_WORDS:
            parts.append(_ORDER_WORDS[token])
            continue
        if token.isdigit():
            parts.append(token)
            continue
        embedded = re.sub(r"\D", "", token)
        if embedded:
            parts.append(embedded)
    return parts


def _order_digit_range() -> tuple[int, int]:
    min_d = _as_int(get_agent_setting("order_id_min_digits", 3), 3, min_value=3, max_value=9)
    max_d = _as_int(get_agent_setting("order_id_max_digits", 6), 6, min_value=min_d, max_value=9)
    return min_d, max_d


def _normalize_order_id_strict(raw_text: str) -> Optional[str]:
    min_d, max_d = _order_digit_range()
    joined = "".join(_extract_digit_parts(raw_text or ""))
    if min_d <= len(joined) <= max_d:
        return joined
    normalized = (raw_text or "").strip().lower()
    matches = re.findall(rf"\d{{{min_d},{max_d}}}", normalized)
    return matches[-1] if matches else None


def _normalize_phone_for_lookup(raw_text: str) -> Optional[str]:
    digits = "".join(_extract_digit_parts(raw_text or ""))
    if not digits:
        return None
    min_digits = _as_int(get_agent_setting("phone_lookup_min_digits", 10), 10, min_value=7, max_value=15)
    max_digits = _as_int(get_agent_setting("phone_lookup_max_digits", 15), 15, min_value=min_digits, max_value=15)
    rx = str(get_agent_setting("phone_lookup_regex", r"^\d{10,15}$") or "").strip()
    if rx:
        try:
            if re.fullmatch(rx, digits):
                return digits
        except re.error:
            pass
    if min_digits <= len(digits) <= max_digits:
        return digits
    return None


def _mentions_no_order_number(text: str) -> bool:
    t = (text or "").lower()
    return bool(
        re.search(r"(δεν έχω|δεν το έχω|χωρίς αριθμό|δεν τον έχω)", t)
        or re.search(r"(δεν έλαβα|δεν πήρα).*(email|επιβεβαίωση)", t)
    )


def _mentions_phone_lookup_intent(text: str) -> bool:
    t = (text or "").lower()
    return bool(
        re.search(r"\b(τηλέφωνο|κινητό|αριθμό|καλέστε με στο)\b", t)
        or re.search(r"(έλεγχος με τηλέφωνο|με το τηλέφωνο)", t)
    )


def _is_order_relevant(text: str) -> bool:
    """Check if the text is likely an attempt to provide order info or ask about it."""
    t = (text or "").lower()
    # If it has any digits, it's likely an attempt at a number
    if re.search(r"\d", t):
        return True
    # Keywords that suggest they are still in the flow
    keywords = {
        "παραγγελία", "αριθμό", "τηλέφωνο", "έλεγχος", "βρες", "βρείτε", "πού", "που",
        "πρόβλημα", "θέμα", "id", "βοήθεια", "ναι", "εντάξει", "έτοιμος", "έτοιμη",
        "order", "number", "phone", "help", "yes", "ok", "ready"
    }
    tokens = set(re.findall(r"\w+", t))
    return bool(tokens & keywords)


# -----------------------------------------------------------------------------
# Room logs and state
# -----------------------------------------------------------------------------


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", value or "")


def _create_room_logger(room_name: str, job_id: Optional[str]) -> tuple[logging.Logger, str]:
    log_dir = os.getenv("ROOM_LOG_DIR", "./data/room-logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"room_{_safe_slug(room_name)}_{_safe_slug(job_id or 'job')}_{ts}.log"
    path = os.path.join(log_dir, filename)

    room_logger = logging.getLogger(f"room.{_safe_slug(room_name)}.{_safe_slug(job_id or 'job')}.{ts}")
    room_logger.setLevel(logging.INFO)
    room_logger.propagate = False
    if not room_logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)sZ | %(levelname)s | %(message)s")
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        room_logger.addHandler(handler)
    return room_logger, path


@dataclass
class SessionState:
    support_state: str = "idle"  # idle|awaiting_order|checking_order|awaiting_phone|checking_phone|ticket_name|ticket_phone|ticket_email|ticket_issue|ticket_confirm|creating_ticket
    ui_state: str = "idle"  # idle|listening|thinking|speaking
    last_issue: str = ""
    last_order_number: str = ""
    last_phone_number: str = ""
    lookup_inflight: bool = False
    ticket_inflight: bool = False
    ticket_name: str = ""
    ticket_phone: str = ""
    ticket_email: str = ""
    ticket_issue: str = ""
    should_end: bool = False
    disconnect_reason: str = "session_end"
    silence_enabled: bool = True
    silence_timeout_s: float = 12.0
    silence_max_prompts: int = 2
    silence_prompt_count: int = 0
    silence_snooze_until: float = 0.0
    waiting_for_user: bool = False
    last_user_activity: float = 0.0
    last_agent_activity: float = 0.0
    last_user_transcript_text: str = ""
    last_user_transcript_at: float = 0.0
    last_agent_transcript_text: str = ""
    last_agent_transcript_at: float = 0.0
    last_clarification_prompt_text: str = ""
    last_clarification_prompt_at: float = 0.0
    suppress_llm_until: float = 0.0
    last_user_text_norm: str = ""
    user_repetition_count: int = 0


_current = {
    "room_logger": None,
    "room_name": None,
    "job_id": None,
    "call_id": None,
    "state": SessionState(),
}


def room_log(event: str, **fields):
    rl = _current.get("room_logger")
    if not rl:
        return
    payload = {
        "event": event,
        "room": _current.get("room_name"),
        "job_id": _current.get("job_id"),
        "call_id": _current.get("call_id"),
    }
    payload.update(fields)
    rl.info(json.dumps(payload, ensure_ascii=False))


# -----------------------------------------------------------------------------
# DB call lifecycle helpers
# -----------------------------------------------------------------------------


async def record_call_to_db(room_name: str, call_type: str = "web", caller_number: str = None, caller_identity: str = None) -> Optional[str]:
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        return await db.record_call_start(
            room_name=room_name,
            call_type=call_type,
            caller_number=caller_number,
            caller_identity=caller_identity,
        )
    except Exception as e:
        logger.warning("Failed to record call start: %s", e)
        return None


async def end_call_in_db(call_id: str = None, room_name: str = None, status: str = "completed", duration_seconds: int = None, disconnect_reason: str = None, transcript: str = None) -> bool:
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        return await db.record_call_end(
            call_id=call_id,
            room_name=room_name,
            status=status,
            duration_seconds=duration_seconds,
            disconnect_reason=disconnect_reason,
            transcript=transcript,
        )
    except Exception as e:
        logger.warning("Failed to record call end: %s", e)
        return False


async def save_transcript_to_db(call_id: str, text: str, speaker: str = "agent", append: bool = True) -> bool:
    if not call_id or not text:
        return False
    try:
        from src.services.database import get_database_service
        db = get_database_service()
        transcript = f"{speaker.capitalize()}: {text}" if append else text
        return await db.update_call_transcript(call_id=call_id, transcript=transcript, append=append)
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


class ElenaFunctionContext(llm.FunctionContext):
    @llm.ai_callable()
    async def lookup_order(self, order_number: Annotated[str, llm.TypeInfo(description="Order number")]) -> str:
        """Look up an order by order number and return a customer-facing summary."""
        room_log("TOOL_CALL", name="lookup_order", order_number=order_number)
        result = await order_lookup.lookup_order(order_number)
        room_log("TOOL_RESULT", name="lookup_order", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def get_order_details(self, order_number: Annotated[str, llm.TypeInfo(description="Order number or last")] = "last") -> str:
        """Fetch detailed information for a specific order or the last looked-up order."""
        room_log("TOOL_CALL", name="get_order_details", order_number=order_number)
        result = await order_lookup.get_order_details(order_number)
        room_log("TOOL_RESULT", name="get_order_details", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def lookup_order_by_phone(self, phone: Annotated[str, llm.TypeInfo(description="Phone number")]) -> str:
        """Look up the most relevant order using the customer's phone number."""
        room_log("TOOL_CALL", name="lookup_order_by_phone", phone=phone)
        result = await order_lookup.lookup_order_by_phone(phone)
        room_log("TOOL_RESULT", name="lookup_order_by_phone", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def search_knowledge_base(self, query: Annotated[str, llm.TypeInfo(description="Question to search")]) -> str:
        """Search the knowledge base for policy, menu, and general support answers."""
        room_log("TOOL_CALL", name="search_knowledge_base", query=query)
        result = await knowledge_base.search_knowledge_base(query, language="el")
        room_log("TOOL_RESULT", name="search_knowledge_base", result=_truncate(result))
        return result

    @llm.ai_callable()
    async def create_support_ticket(
        self,
        customer_name: Annotated[str, llm.TypeInfo(description="Customer full name")],
        customer_phone: Annotated[str, llm.TypeInfo(description="Customer phone")],
        customer_email: Annotated[str, llm.TypeInfo(description="Customer email")],
        issue_description: Annotated[str, llm.TypeInfo(description="Issue")],
    ) -> str:
        """Create a support ticket when an issue cannot be resolved during the call."""
        room_log("TOOL_CALL", name="create_support_ticket")
        result = await support_ticket.create_support_ticket(
            customer_name,
            customer_phone,
            customer_email,
            issue_description,
        )
        room_log("TOOL_RESULT", name="create_support_ticket", result=_truncate(result))

        # Sync deterministic state: if LLM created a ticket, we are no longer in a support flow.
        state = _current["state"]
        state.support_state = "idle"
        state.ticket_name = ""
        state.ticket_phone = ""
        state.ticket_email = ""
        state.ticket_issue = ""

        return result

    @llm.ai_callable()
    async def end_session(self) -> str:
        """
        End the voice call session gracefully.
        Use this when the customer says goodbye, thanks, or indicates they're done.

        Examples of when to use:
        - "Goodbye", "Bye", "Thanks, bye"
        - "That's all I needed", "I'm done"
        - "Have a nice day", "Thank you, that's it"

        Returns:
            Goodbye message - you MUST speak this message, the call will end after
        """
        logger.info("Session end requested - scheduling disconnect after goodbye")
        room_log("SESSION_END_REQUESTED")

        # Disable silence monitor immediately!
        _current["state"].silence_enabled = False

        # Schedule the disconnect with a delay to allow goodbye to be spoken
        async def delayed_end():
            # Wait for LLM to process response + TTS to generate + speak
            # This needs to be long enough for the full goodbye to be heard
            await asyncio.sleep(6.0)  # 6 seconds should be plenty
            _current["state"].should_end = True
            logger.info("Delayed session end triggered")

        asyncio.create_task(delayed_end())

        # Return closing message
        goodbye = get_closing("el")
        room_log("SESSION_END_MESSAGE", text=_truncate(goodbye))
        return goodbye


# -----------------------------------------------------------------------------
# Providers
# -----------------------------------------------------------------------------


def create_llm():
    model = str(get_agent_setting("llm_model", "gpt-4o-mini") or "gpt-4o-mini")
    return openai.LLM(model=model, api_key=settings.openai_api_key)


def create_stt(is_sip_call: bool = False):
    provider = str(get_agent_setting("stt_provider", "deepgram") or "deepgram").lower()
    deepgram_api_key = getattr(settings, "deepgram_api_key", None)
    if provider == "deepgram" and USE_DEEPGRAM and deepgram_api_key:
        # Use the model configured in the DB, defaulting to nova-3 (same as English).
        model = str(get_agent_setting("deepgram_stt_model", "nova-3") or "nova-3").strip()

        # Base config matching elena_original.py / elena_en.py style — smart_format=True
        # is critical for reliable digit transcription (e.g. "επτά επτά" → "77").
        base = {
            "model": model,
            "language": "el",
            "interim_results": True,
            "punctuate": True,
            "smart_format": True,
        }

        # Try with explicit api_key first; fall back without it (some SDK versions
        # reject api_key as an unknown kwarg).
        for kwargs in [
            {**base, "api_key": deepgram_api_key},
            base,
        ]:
            try:
                logger.info(
                    "Creating Deepgram STT (EL): model=%s language=el smart_format=True api_key_passed=%s",
                    model,
                    "api_key" in kwargs,
                )
                return deepgram.STT(**kwargs)
            except TypeError as e:
                logger.warning("Deepgram STT args not supported, retrying with fallback args: %s", e)
                continue

        # Hard fallback: minimal args only
        return deepgram.STT(model=model, language="el")

    openai_model = str(get_agent_setting("openai_stt_model", "whisper-1") or "whisper-1")
    return openai.STT(model=openai_model, api_key=settings.openai_api_key, language="el")


def create_tts():
    provider = str(get_agent_setting("tts_provider", "elevenlabs") or "elevenlabs").lower()
    if provider == "openai":
        return openai.TTS(
            model=str(get_agent_setting("openai_tts_model", "tts-1") or "tts-1"),
            voice=str(get_agent_setting("openai_tts_voice", "alloy") or "alloy"),
            speed=_as_float(get_agent_setting("openai_tts_speed", 1.0), 1.0, min_value=0.25, max_value=4.0),
            api_key=settings.openai_api_key,
        )

    # ElevenLabs default (SDK-compatible signature), with safe fallback.
    eleven_api_key = getattr(settings, "elevenlabs_api_key", None)
    if not eleven_api_key:
        logger.warning("ELEVENLABS_API_KEY missing, falling back to OpenAI TTS")
        return openai.TTS(
            model=str(get_agent_setting("openai_tts_model", "tts-1") or "tts-1"),
            voice=str(get_agent_setting("openai_tts_voice", "alloy") or "alloy"),
            speed=_as_float(get_agent_setting("openai_tts_speed", 1.0), 1.0, min_value=0.25, max_value=4.0),
            api_key=settings.openai_api_key,
        )

    voice_id = str(
        get_agent_setting("agent_voice_id", getattr(settings, "elevenlabs_voice_id", "")) or getattr(settings, "elevenlabs_voice_id", "")
    )
    similarity = _as_float(
        get_agent_setting("agent_voice_similarity", getattr(settings, "elevenlabs_voice_similarity", 0.9)),
        0.9,
        min_value=0.0,
        max_value=1.0,
    )
    stability = _as_float(
        get_agent_setting("agent_voice_stability", getattr(settings, "elevenlabs_voice_stability", 0.65)),
        0.65,
        min_value=0.0,
        max_value=1.0,
    )
    model = str(
        get_agent_setting("elevenlabs_model", getattr(settings, "elevenlabs_model", "eleven_turbo_v2_5") or "eleven_turbo_v2_5")
        or "eleven_turbo_v2_5"
    )

    try:
        voice = elevenlabs.Voice(
            id=voice_id,
            name="Elena",
            category="premade",
            settings=elevenlabs.VoiceSettings(stability=stability, similarity_boost=similarity),
        )
        return elevenlabs.TTS(
            voice=voice,
            model=model,
        )
    except TypeError as e:
        logger.warning("ElevenLabs TTS init failed (%s), falling back to OpenAI TTS", e)
        return openai.TTS(
            model=str(get_agent_setting("openai_tts_model", "tts-1") or "tts-1"),
            voice=str(get_agent_setting("openai_tts_voice", "alloy") or "alloy"),
            speed=_as_float(get_agent_setting("openai_tts_speed", 1.0), 1.0, min_value=0.25, max_value=4.0),
            api_key=settings.openai_api_key,
        )


def _looks_like_email(text: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", (text or "").strip()))


def _is_yes(text: str) -> bool:
    t = (text or "").strip().lower()
    # Flexible match for common Greek affirmative phrases
    keywords = {"ναι", "σωστά", "επιβεβαιώνω", "προχώρα", "μάλιστα", "εντάξει", "οκ", "ναι σωστά", "κάνε", "δημιούργησε"}
    return any(w in t for w in keywords)


def _is_no(text: str) -> bool:
    t = (text or "").strip().lower()
    # Flexible match for common Greek negative phrases
    keywords = {"όχι", "μη", "ακύρωση", "σταμάτα", "ποτέ", "όχι τώρα"}
    return any(w in t for w in keywords)


def _normalize_intent_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a0-9\s]", " ", (text or "").lower())).strip()


def _intent_tokens(text: str) -> set[str]:
    toks = _normalize_intent_text(text).split()
    normalized: set[str] = set()
    for tok in toks:
        if len(tok) < 3 or tok in _MEMORY_STOPWORDS:
            continue
        t = tok
        # Generic light stemming (language-agnostic enough for EN, no domain hardcoding).
        if t.endswith("ies") and len(t) > 4:
            t = t[:-3] + "y"
        elif t.endswith("es") and len(t) > 4:
            t = t[:-2]
        elif t.endswith("s") and len(t) > 4:
            t = t[:-1]
        elif t.endswith("ing") and len(t) > 5:
            t = t[:-3]
        elif t.endswith("ed") and len(t) > 4:
            t = t[:-2]
        normalized.add(t)
    return normalized


def _find_memory_match(user_text: str, memory_items: list[dict]) -> Optional[str]:
    user_norm = _normalize_intent_text(user_text)
    user_tokens = _intent_tokens(user_text)
    if not user_norm or not user_tokens:
        return None

    best_answer: Optional[str] = None
    best_score = 0.0

    for item in memory_items:
        q = str(item.get("question") or "").strip()
        a = str(item.get("answer") or "").strip()
        if not q or not a:
            continue
        q_norm = _normalize_intent_text(q)
        q_tokens = _intent_tokens(q)
        if not q_tokens:
            continue

        overlap = len(user_tokens & q_tokens)
        coverage = overlap / max(1, len(q_tokens))
        recall = overlap / max(1, len(user_tokens))
        phrase_hit = 1.0 if (q_norm in user_norm or user_norm in q_norm) else 0.0
        score = (coverage * 0.55) + (recall * 0.35) + (phrase_hit * 0.25)

        if score > best_score and (overlap >= 2 or phrase_hit > 0):
            best_score = score
            best_answer = a

    return best_answer if best_score >= 0.40 else None


def _build_memory_prompt_block(memory_items: list[dict]) -> str:
    lines: list[str] = []
    for item in memory_items:
        q = str(item.get("question") or "").strip()
        a = str(item.get("answer") or "").strip()
        c = str(item.get("comment") or item.get("comments") or "").strip()
        if not q or not a:
            continue
        lines.append(f'SCENARIO (match by intent, not exact words): "{q}"')
        lines.append(f'EXPECTED RESPONSE: "{a}"')
        if c:
            lines.append(f"GUIDELINE: {c}")
        lines.append("-" * 20)
    if not lines:
        return ""
    return (
        "### CRITICAL: LONG-TERM MEMORY (HIGHEST PRIORITY)\n"
        "When user intent matches any memory scenario, respond using that memory response first.\n"
        "Treat scenario matching as semantic/intention-based (not exact wording).\n\n"
        "### ABSOLUTE VERBAL GUARDRAILS (INTERNAL ONLY)\n"
        "- NEVER speak section headers (e.g., '## Closing', '### CORE BEHAVIOR', 'GUIDELINE').\n"
        "- NEVER speak behavioral instructions or meta-rules (e.g., phrases starting with 'Never say...', 'Always say...', 'Avoid...', 'Close naturally...').\n"
        "- Bullet points in the 'CORE BEHAVIOR' or 'Closing' sections are for your logic reasoning ONLY.\n"
        "- NEVER speak the internal tags: 'SCENARIO', 'EXPECTED RESPONSE', or 'GUIDELINE'.\n"
        "- ONLY speak the actual conversational text intended for the customer.\n\n"
        + "\n".join(lines)
    )


def create_vad(is_sip_call: bool = False):
    min_speech = _as_float(get_agent_setting("vad_min_speech_duration", 0.15), 0.15, min_value=0.05, max_value=1.0)
    min_silence = _as_float(get_agent_setting("vad_min_silence_duration", 1.2 if is_sip_call else 1.0), 1.0, min_value=0.1, max_value=3.0)
    # threshold=0.6: slightly more aggressive to filter out background noise
    return silero.VAD.load(min_speech_duration=min_speech, min_silence_duration=min_silence, activation_threshold=0.6)


# -----------------------------------------------------------------------------
# Core flow helpers
# -----------------------------------------------------------------------------


async def _is_order_not_found_text(text: str) -> bool:
    t = (text or "").lower()
    return "δεν βρέθηκε" in t or "δεν υπάρχει" in t or "no order" in t or "could not find" in t


async def _publish_transcript(ctx: JobContext, speaker: str, text: str):
    cleaned = (text or "").strip()
    if not cleaned:
        return
    payload = json.dumps({"type": "transcript", "speaker": speaker, "text": cleaned}, ensure_ascii=False)
    try:
        await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
    except Exception:
        pass


async def _publish_state(ctx: JobContext, state_name: str):
    payload = json.dumps({"type": "state", "state": state_name}, ensure_ascii=False)
    try:
        await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
    except Exception:
        pass


async def _run_order_lookup(agent: VoicePipelineAgent, order_number: str):
    state: SessionState = _current["state"]
    if state.lookup_inflight:
        return
    state.lookup_inflight = True
    state.support_state = "checking_order"
    room_log("ORDER_LOOKUP_STARTED", order_number=order_number)
    try:
        result = await order_lookup.lookup_order(order_number)
        room_log("ORDER_LOOKUP_RESULT", result=_truncate(result))
        
        # Clean up text to prevent awkward TTS pauses on punctuation/colons/brackets
        from src.utils.voice_formatting import clean_text_for_tts
        cleaned_result = clean_text_for_tts(result, lang="el")
        await agent.say(cleaned_result, allow_interruptions=True)
        
        state.last_order_number = order_number
        if _is_order_not_found_text(result):
            state.support_state = "awaiting_order"
        else:
            state.support_state = "idle"
    finally:
        state.lookup_inflight = False


async def _run_phone_lookup(agent: VoicePipelineAgent, phone_number: str):
    state: SessionState = _current["state"]
    if state.lookup_inflight:
        return
    state.lookup_inflight = True
    state.support_state = "checking_phone"
    room_log("PHONE_LOOKUP_STARTED", phone=phone_number)
    try:
        result = await order_lookup.lookup_order_by_phone(phone_number)
        room_log("PHONE_LOOKUP_RESULT", result=_truncate(result))
        
        # Clean up text to prevent awkward TTS pauses on punctuation/colons/brackets
        from src.utils.voice_formatting import clean_text_for_tts
        cleaned_result = clean_text_for_tts(result, lang="el")
        await agent.say(cleaned_result, allow_interruptions=True)
        
        state.last_phone_number = phone_number
        if "no order" in (result or "").lower() or "could not" in (result or "").lower():
            state.support_state = "awaiting_phone"
        else:
            state.support_state = "idle"
    finally:
        state.lookup_inflight = False


async def _run_create_ticket(agent: VoicePipelineAgent):
    state: SessionState = _current["state"]
    if state.ticket_inflight:
        return
    state.ticket_inflight = True
    state.support_state = "creating_ticket"
    room_log("TICKET_CREATE_STARTED")
    try:
        result = await support_ticket.create_support_ticket(
            state.ticket_name or "Πελάτης",
            state.ticket_phone,
            state.ticket_email,
            state.ticket_issue,
        )
        room_log("TICKET_CREATE_RESULT", result=_truncate(result))
        await say(result, allow_interruptions=True)
        state.support_state = "idle"
        state.last_issue = ""  # clear so silence monitor uses generic prompt, not "still here to help"
        state.ticket_name = ""
        state.ticket_phone = ""
        state.ticket_email = ""
        state.ticket_issue = ""
    finally:
        state.ticket_inflight = False


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


async def entrypoint(ctx: JobContext):
    set_runtime_language("el")
    # Warm cache in background; do not block first response path on DB latency.
    cache_task = asyncio.create_task(_fetch_from_db())

    state = SessionState(
        silence_timeout_s=_as_float(get_agent_setting("silence_timeout_seconds", 5.0), 5.0, min_value=1.0, max_value=60.0),
        silence_max_prompts=_as_int(get_agent_setting("silence_max_prompts", 0), 0, min_value=0, max_value=5),
        last_user_activity=time.time(),
        last_agent_activity=time.time(),
    )
    _current["state"] = state
    memory_items: list[dict] = []

    async def set_ui_state(new_state: str):
        if state.ui_state == new_state:
            return
        state.ui_state = new_state
        room_log("UI_STATE", state=new_state)
        await _publish_state(ctx, new_state)

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
                await set_ui_state("thinking")
            except asyncio.CancelledError:
                return

        thinking_task = asyncio.create_task(_set_thinking())

    def snooze_silence(seconds: float):
        now_ts = time.time()
        state.silence_snooze_until = max(state.silence_snooze_until, now_ts + max(0.0, seconds))
        room_log("SILENCE_SNOOZE", until=state.silence_snooze_until, seconds=seconds)

    def _should_suppress_clarification(text: str, min_gap_s: float = 6.0) -> bool:
        now_ts = time.time()
        normalized = " ".join((text or "").strip().lower().split())
        if not normalized:
            return True
        if (
            normalized == state.last_clarification_prompt_text
            and (now_ts - state.last_clarification_prompt_at) < max(0.0, min_gap_s)
        ):
            room_log("CLARIFICATION_SUPPRESSED", text=normalized, min_gap_s=min_gap_s)
            return True
        state.last_clarification_prompt_text = normalized
        state.last_clarification_prompt_at = now_ts
        return False

    job = getattr(ctx, "job", None)
    job_id = getattr(job, "id", None) or getattr(job, "job_id", None)
    room_logger, room_log_path = _create_room_logger(ctx.room.name, job_id)
    _current["room_logger"] = room_logger
    _current["room_name"] = ctx.room.name
    _current["job_id"] = job_id
    room_log("ROOM_START", call_type="web")
    logger.info("Per-room log: %s", room_log_path)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    room_log("PARTICIPANT_CONNECTED", identity=participant.identity)

    call_id = await record_call_to_db(ctx.room.name, call_type="web", caller_identity=participant.identity)
    _current["call_id"] = call_id

    try:
        await asyncio.wait_for(cache_task, timeout=8.0)
    except Exception as e:
        logger.warning("Initial cache warmup did not complete in time: %s", e)

    try:
        from src.services.database import get_database_service
        db = get_database_service()
        memory_items = await db.get_memory_items(active_only=True)
        room_log("MEMORY_ITEMS_LOADED", count=len(memory_items))
    except Exception as e:
        logger.warning("Failed loading memory items for direct matcher: %s", e)
        room_log("MEMORY_ITEMS_LOAD_FAILED", error=str(e))

    try:
        system_prompt = await get_system_prompt_async("el")
    except Exception as e:
        logger.warning("System prompt load failed, using fallback prompt: %s", e)
        system_prompt = (
            "Είστε η Elena από την Meallion. Απαντήστε σε σύντομα, φιλικά Ελληνικά. "
            "Για θέματα παραγγελιών, ζητήστε πρώτα τον αριθμό παραγγελίας και μετά το τηλέφωνο αν χρειαστεί."
        )
    # Ensure memory guidance is always present even if prompts cache misses memory context.
    memory_block = _build_memory_prompt_block(memory_items)
    if memory_block and "CRITICAL: LONG-TERM MEMORY" not in (system_prompt or ""):
        system_prompt = f"{memory_block}\n\n{system_prompt}"
    has_memory_block = "LONG-TERM MEMORY" in (system_prompt or "")
    room_log("SYSTEM_PROMPT_READY", length=len(system_prompt or ""), has_memory_block=has_memory_block)
    chat_ctx = llm.ChatContext()
    chat_ctx.append(role="system", text=system_prompt)

    configured_endpointing_delay = _as_float(
        get_agent_setting("min_endpointing_delay", 1.2),
        1.2,
        min_value=0.2,
        max_value=3.0,
    )
    # Patience: wait at least ~4s after user stops speaking before replying.
    effective_endpointing_delay = max(4.0, configured_endpointing_delay)

    def _before_llm_cb(agent_instance, chat_ctx):
        """Gate the LLM when the deterministic handler has already replied via agent.say()."""
        if time.time() < state.suppress_llm_until:
            room_log("LLM_SUPPRESSED", until=state.suppress_llm_until)
            return False
        from livekit.agents.pipeline.pipeline_agent import _default_before_llm_cb
        return _default_before_llm_cb(agent_instance, chat_ctx)

    agent = VoicePipelineAgent(
        vad=create_vad(),
        stt=create_stt(),
        llm=create_llm(),
        tts=create_tts(),
        chat_ctx=chat_ctx,
        fnc_ctx=ElenaFunctionContext(),
        allow_interruptions=True,
        interrupt_min_words=_as_int(get_agent_setting("interrupt_min_words", 2), 2, min_value=1, max_value=10),
        min_endpointing_delay=effective_endpointing_delay,
        # Keep disabled here to avoid race where LLM starts replying before deterministic
        # memory/order flow handlers finish, which can produce double answers.
        preemptive_synthesis=_as_bool(get_agent_setting("preemptive_synthesis", False), default=False),
        before_llm_cb=_before_llm_cb,
    )

    room_log(
        "TURN_CONFIG",
        configured_endpointing_delay=configured_endpointing_delay,
        effective_endpointing_delay=effective_endpointing_delay,
    )

    conversation_transcript: list[str] = []

    def suppress_llm(seconds: float = 10.0):
        """Suppress LLM synthesis for the next N seconds (used when handler replies deterministically)."""
        state.suppress_llm_until = time.time() + seconds
        room_log("LLM_SUPPRESS_SET", seconds=seconds)

    async def send_agent_transcript(text: str):
        cleaned = (text or "").strip()
        if not cleaned:
            return
        now_ts = time.time()
        if cleaned == state.last_agent_transcript_text and (now_ts - state.last_agent_transcript_at) < 2.5:
            room_log("AGENT_TEXT_DEDUPED", text=cleaned)
            return
        state.last_agent_transcript_text = cleaned
        state.last_agent_transcript_at = now_ts
        conversation_transcript.append(f"Agent: {cleaned}")
        await _publish_transcript(ctx, "agent", cleaned)
        if call_id:
            await save_transcript_to_db(call_id, cleaned, speaker="agent")
            transcript_text = "\n".join(conversation_transcript)
            await save_transcript_to_db(call_id, transcript_text, speaker="full", append=False)
        room_log("AGENT_TEXT", text=cleaned)

    _last_user_interim = ""
    _last_user_interim_sent_at = 0.0

    async def send_user_transcript(text: str, *, interim: bool = False):
        nonlocal _last_user_interim, _last_user_interim_sent_at
        cleaned = (text or "").strip()
        if not cleaned:
            return
        now_ts = time.time()
        if interim:
            # Throttle interim updates to keep UI smooth and avoid flooding.
            if cleaned == _last_user_interim and (now_ts - _last_user_interim_sent_at) < 0.35:
                return
            _last_user_interim = cleaned
            _last_user_interim_sent_at = now_ts
            payload = json.dumps(
                {"type": "transcript", "speaker": "user", "text": cleaned, "interim": True},
                ensure_ascii=False,
            )
            try:
                # Use unreliable delivery for interims to minimize latency.
                await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=False)
            except Exception:
                pass
            return

        if cleaned == state.last_user_transcript_text and (now_ts - state.last_user_transcript_at) < 5.0:
            room_log("USER_TEXT_DEDUPED", text=cleaned)
            return

        state.last_user_transcript_text = cleaned
        state.last_user_transcript_at = now_ts
        _last_user_interim = ""
        conversation_transcript.append(f"User: {cleaned}")
        payload = json.dumps(
            {"type": "transcript", "speaker": "user", "text": cleaned, "interim": False},
            ensure_ascii=False,
        )
        try:
            await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
        except Exception:
            pass
        if call_id:
            await save_transcript_to_db(call_id, cleaned, speaker="user")
            transcript_text = "\n".join(conversation_transcript)
            await save_transcript_to_db(call_id, transcript_text, speaker="full", append=False)
        room_log("USER_TEXT", text=cleaned)

    @agent.on("agent_started_speaking")
    def _on_agent_started_speaking():
        cancel_thinking_task()
        state.waiting_for_user = False
        asyncio.create_task(set_ui_state("speaking"))

    @agent.on("agent_stopped_speaking")
    def _on_agent_stopped_speaking():
        state.last_agent_activity = time.time()
        state.waiting_for_user = True
        asyncio.create_task(set_ui_state("idle"))


    @agent.on("agent_speech_committed")
    def _on_agent_speech_committed(msg):
        text = msg.content if hasattr(msg, "content") else None
        if text:
            asyncio.create_task(send_agent_transcript(text))

    @agent.on("agent_speech_interrupted")
    def _on_agent_speech_interrupted(msg):
        # Fallback: when user barges in before commit, capture whatever text exists.
        text = msg.content if hasattr(msg, "content") else None
        if text:
            room_log("AGENT_TEXT_INTERRUPTED_CAPTURE", text=_truncate(text))
            asyncio.create_task(send_agent_transcript(text))

    @agent.on("user_started_speaking")
    def _on_user_started_speaking():
        cancel_thinking_task()
        state.waiting_for_user = False
        asyncio.create_task(set_ui_state("listening"))

    @agent.on("user_stopped_speaking")
    def _on_user_stopped_speaking():
        schedule_thinking_state()

    @agent.on("user_speech_committed")
    def _on_user_speech_committed(msg):
        user_text = str(getattr(msg, "content", "") or "").strip()
        if not user_text:
            return

        state.last_user_activity = time.time()
        state.silence_prompt_count = 0
        asyncio.create_task(send_user_transcript(user_text))

        # 0a) Repetition check: if user repeats the same sentence 2 times, disconnect.
        # Normalize: remove non-alphanumeric and convert to lowercase
        norm_text = re.sub(r"[^a-z0-9α-ωά-ώ]", "", user_text.lower())
        if norm_text and norm_text == state.last_user_text_norm:
            state.user_repetition_count += 1
            if state.user_repetition_count >= 1: # 1 repetition = 2 times total
                room_log("REPETITION_TERMINATION", text=user_text)
                state.silence_enabled = False # Stop silence monitor immediately
                asyncio.create_task(agent.say("Το έχω ακούσει ήδη αυτό. Θα κλείσω την κλήση τώρα. Γεια σας!", allow_interruptions=True))
                async def _end_rep():
                    await asyncio.sleep(5.0) # Wait for agent to start/finish speaking
                    state.should_end = True
                    state.disconnect_reason = "repetition_termination"
                asyncio.create_task(_end_rep())
                return
        else:
            state.last_user_text_norm = norm_text
            state.user_repetition_count = 0

        # 0b) Farewell intent (Broadened and Hoisted to Step 0)
        # Check for goodbye keywords or common close phrases.
        farewell_intent = bool(re.search(
            r"\b(αντίο|καληνύχτα|γεια|ευχαριστώ|ευχαριστούμε|αυτά μόνο|τίποτα άλλο|όχι ευχαριστώ|κλείσε|τελειώσαμε|όλα καλά|αυτό ήταν)\b",
            user_text.lower()
        ))
        # Composite intent: short sentence with both 'thanks' and 'no' or 'close'
        if not farewell_intent and len(user_text.split()) <= 10:
            has_thanks = bool(re.search(r"(ευχαριστώ|ευχαριστούμε|thx|thanks)", user_text.lower()))
            has_close = bool(re.search(r"(όχι|όχι άλλο|τίποτα|αυτά|γεια|bye|no)", user_text.lower()))
            if has_thanks and has_close:
                farewell_intent = True

        # Ignore farewell if user is providing a valid-looking number
        if farewell_intent and len(re.findall(r"\d", user_text)) >= 3:
            farewell_intent = False

        if farewell_intent:
            room_log("FAREWELL_DETECTED", text=user_text)
            state.silence_enabled = False
            suppress_llm(15.0)
            
            async def _delayed_end_farewell():
                # Say a deterministic goodbye to ensure closure even if LLM is slow
                await agent.say("Παρακαλώ! Χαίρομαι που βοήθησα. Καλή σας ημέρα!", allow_interruptions=True)
                await asyncio.sleep(2.0)
                state.should_end = True
                state.disconnect_reason = "farewell"
            asyncio.create_task(_delayed_end_farewell())
            return

        # 1) If lookup is in progress, keep caller informed and do not branch.
        if state.lookup_inflight:
            asyncio.create_task(set_ui_state("thinking"))
            snooze_silence(8.0)
            asyncio.create_task(agent.say("Το ελέγχω αυτή τη στιγμή. Μισό λεπτό παρακαλώ.", allow_interruptions=True))
            return

        if state.ticket_inflight:
            asyncio.create_task(set_ui_state("thinking"))
            snooze_silence(8.0)
            asyncio.create_task(agent.say("Δημιουργώ το αίτημα υποστήριξης τώρα. Μισό λεπτό παρακαλώ.", allow_interruptions=True))
            return

        # 1.5) Ticket-creation escape — checked before any order/phone state branches.
        _ticket_escape = bool(re.search(
            r"\b(άνθρωπο|εκπρόσωπο|υπάλληλο|καλέστε με|επικοινωνία|αίτημα υποστήριξης|ticket|άνοιγμα ticket|δημιουργία ticket|παράπονο)\b",
            user_text.lower()
        ))
        _in_ticket_flow = state.support_state in {
            "ticket_name", "ticket_phone", "ticket_email",
            "ticket_issue", "ticket_confirm", "creating_ticket"
        }
        if _ticket_escape and not _in_ticket_flow:
            room_log("FLOW_TRANSITION", from_state=state.support_state, to_state="ticket_name", reason="ticket_escape")
            state.support_state = "ticket_name"
            suppress_llm(15.0)
            asyncio.create_task(agent.say(
                "Βεβαίως, μπορώ να δημιουργήσω ένα αίτημα υποστήριξης. Παρακαλώ πείτε μου το πλήρες όνομά σας.",
                allow_interruptions=True
            ))
            return

        # 2) Active order-support flow
        if state.support_state in {"awaiting_order", "checking_order"}:
            # If user indicates phone lookup path, move flow to phone collection.
            if _mentions_no_order_number(user_text) or _mentions_phone_lookup_intent(user_text):
                state.support_state = "awaiting_phone"
                room_log("FLOW_TRANSITION", from_state="awaiting_order", to_state="awaiting_phone", reason="no_order_or_phone_intent")
                # Let the LLM handle the transition sentence naturally.
                return

            # If user already gave a full phone-like number, use phone lookup directly.
            phone_candidate = _normalize_phone_for_lookup(user_text)
            if phone_candidate:
                state.support_state = "awaiting_phone"
                suppress_llm(15.0)
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(15.0)
                asyncio.create_task(_run_phone_lookup(agent, phone_candidate))
                return

            order_id = _normalize_order_id_strict(user_text)
            if order_id:
                suppress_llm(15.0)
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(15.0)
                asyncio.create_task(_run_order_lookup(agent, order_id))
                return

            # If user is talking about order but didn't give valid ID, give deterministic hint.
            if _is_order_relevant(user_text):
                prompt = "Όποτε είστε έτοιμοι, παρακαλώ πείτε μου τον αριθμό παραγγελίας σας. Αν δεν τον έχετε, πείτε το και θα τον βρω με το τηλέφωνό σας."
                if not _should_suppress_clarification(prompt):
                    suppress_llm()
                    asyncio.create_task(agent.say(prompt, allow_interruptions=True))
                return

            # Let LLM handle potential diversions
            return

        # 3) Active phone-support flow
        if state.support_state in {"awaiting_phone", "checking_phone"}:
            if _mentions_phone_lookup_intent(user_text):
                prompt = "Βεβαίως. Παρακαλώ δώστε μου το πλήρες τηλέφωνο που χρησιμοποιήσατε για την παραγγελία."
                if not _should_suppress_clarification(prompt):
                    suppress_llm()
                    asyncio.create_task(agent.say(prompt, allow_interruptions=True))
                return
            phone = _normalize_phone_for_lookup(user_text)
            if phone:
                suppress_llm(15.0)
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(15.0)
                asyncio.create_task(_run_phone_lookup(agent, phone))
                return
            
            # Escape: user may give an order ID instead of a phone number.
            order_id_escape = _normalize_order_id_strict(user_text)
            if order_id_escape:
                room_log("FLOW_TRANSITION", from_state="awaiting_phone", to_state="awaiting_order", reason="order_id_given_in_phone_flow")
                state.support_state = "awaiting_order"
                suppress_llm(15.0)
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(15.0)
                asyncio.create_task(_run_order_lookup(agent, order_id_escape))
                return

            # Only clarify if relevant to order flow
            if _is_order_relevant(user_text):
                prompt = "Χρειάζομαι το πλήρες τηλέφωνο για να βρω την παραγγελία. Παρακαλώ πείτε το."
                if not _should_suppress_clarification(prompt):
                    suppress_llm()
                    asyncio.create_task(agent.say(prompt, allow_interruptions=True))
                return

            # Fall through for diversions
            return

        # 3b) Support ticket flow
        if state.support_state.startswith("ticket_"):
            # Stronger LLM suppression while we are in a deterministic sub-flow
            suppress_llm(15.0)
        if state.support_state == "ticket_name":
            state.ticket_name = user_text
            state.support_state = "ticket_phone"
            suppress_llm()
            asyncio.create_task(agent.say("Ευχαριστώ. Παρακαλώ πείτε μου το τηλέφωνό σας.", allow_interruptions=True))
            return

        if state.support_state == "ticket_phone":
            ticket_phone = _normalize_phone_for_lookup(user_text)
            if not ticket_phone:
                prompt = "Παρακαλώ δώστε έναν έγκυρο αριθμό τηλεφώνου."
                if not _should_suppress_clarification(prompt):
                    suppress_llm()
                    asyncio.create_task(agent.say(prompt, allow_interruptions=True))
                return
            state.ticket_phone = ticket_phone
            state.support_state = "ticket_email"
            suppress_llm()
            asyncio.create_task(agent.say("Τέλεια. Τώρα παρακαλώ πείτε μου το email σας.", allow_interruptions=True))
            return

        if state.support_state == "ticket_email":
            if not _looks_like_email(user_text):
                suppress_llm()
                asyncio.create_task(agent.say("Παρακαλώ δώστε μια έγκυρη διεύθυνση email.", allow_interruptions=True))
                return
            state.ticket_email = user_text.strip()
            state.support_state = "ticket_issue"
            suppress_llm()
            asyncio.create_task(agent.say("Παρακαλώ περιγράψτε το πρόβλημα με μία ή δύο προτάσεις.", allow_interruptions=True))
            return

        if state.support_state == "ticket_issue":
            state.ticket_issue = user_text
            state.support_state = "ticket_confirm"
            confirm_text = (
                f"Έχω τα στοιχεία σας ως: όνομα {state.ticket_name}, τηλέφωνο {state.ticket_phone}, και email {state.ticket_email}. "
                "Να δημιουργήσω το αίτημα υποστήριξης τώρα;"
            )
            suppress_llm()
            asyncio.create_task(agent.say(confirm_text, allow_interruptions=True))
            return

        if state.support_state == "ticket_confirm":
            if _is_yes(user_text):
                suppress_llm()
                asyncio.create_task(set_ui_state("thinking"))
                snooze_silence(10.0)
                asyncio.create_task(agent.say("Ευχαριστώ. Δημιουργώ το αίτημα τώρα.", allow_interruptions=True))
                asyncio.create_task(_run_create_ticket(agent))
                return
            if _is_no(user_text):
                state.support_state = "idle"
                state.ticket_name = ""
                state.ticket_phone = ""
                state.ticket_email = ""
                state.ticket_issue = ""
                suppress_llm()
                asyncio.create_task(agent.say("Κανένα πρόβλημα. Ακύρωσα το αίτημα.", allow_interruptions=True))
                return
            suppress_llm()
            asyncio.create_task(agent.say("Παρακαλώ πείτε ναι για δημιουργία ή όχι για ακύρωση.", allow_interruptions=True))
            return

        # General turno logic is handled by LLM fallthrough above.
        return

    # Optional: capture agent text word-by-word if needed, but committed is safer for translations.
    # We already have _on_agent_speech_committed.
    

        # 5) Otherwise let LLM handle general query naturally.

    # Participant disconnect
    @ctx.room.on("participant_disconnected")
    def _on_participant_disconnected(participant_info):
        if participant_info.identity == participant.identity:
            state.should_end = True
            state.disconnect_reason = "participant_disconnected"

    # Start agent
    agent.start(ctx.room, participant)

    # Stream interim user transcripts so user text appears in real time on frontend.
    human_input = getattr(agent, "human_input", None) or getattr(agent, "_human_input", None)
    room_log("HUMAN_INPUT_ATTACH", found=bool(human_input))
    if human_input:
        @human_input.on("interim_transcript")
        def _on_interim_transcript(ev):
            try:
                # ev is a SpeechEvent; extract text from the first alternative
                text = ev.alternatives[0].text if ev.alternatives else None
            except Exception:
                text = None
                
            if text:
                cancel_thinking_task()
                asyncio.create_task(send_user_transcript(text, interim=True))

    # Greet
    greeting_enabled = _as_bool(get_agent_setting("agent_greeting_enabled", True), default=True)
    if greeting_enabled:
        greeting = get_greeting("el")
        chat_ctx.append(role="assistant", text=greeting)
        await agent.say(greeting, allow_interruptions=True)
    await set_ui_state("idle")

    # Start background audio after greeting so first response is not delayed.
    bg_audio_player = None

    async def _start_background_audio():
        nonlocal bg_audio_player
        try:
            from src.services.background_audio import create_background_audio_player
            bg_audio_player = await create_background_audio_player()
            if bg_audio_player:
                ok = await bg_audio_player.start(ctx.room)
                room_log("BG_AUDIO_START", enabled=True, started=bool(ok))
            else:
                room_log("BG_AUDIO_START", enabled=False, started=False)
        except Exception as e:
            room_log("BG_AUDIO_START_ERROR", error=str(e))

    bg_audio_task = asyncio.create_task(_start_background_audio())
    bg_audio_task.set_name("bg_audio_init")

    # Prevent immediate silence prompt right after startup/greeting delays.
    now = time.time()
    state.last_user_activity = now
    state.last_agent_activity = now

    # Prefetch orders in background
    asyncio.create_task(order_lookup.prefetch_orders())

    def _contextual_silence_prompt() -> str:
        phase = state.silence_prompt_count
        support_state = state.support_state

        if support_state in {"awaiting_order", "checking_order"}:
            if phase == 0:
                return "Όποτε είστε έτοιμοι, παρακαλώ πείτε μου τον αριθμό παραγγελίας σας. Αν δεν τον έχετε, πείτε το και θα τον βρω με το τηλέφωνό σας."
            return "Είμαι ακόμα εδώ. Παρακαλώ πείτε μου τον αριθμό παραγγελίας σας, ή πείτε ότι δεν τον έχετε για να τον βρω με το τηλέφωνο."

        if support_state in {"awaiting_phone", "checking_phone"}:
            if phase == 0:
                return "Παρακαλώ πείτε μου τον αριθμό τηλεφώνου που χρησιμοποιήσατε για την παραγγελία όταν είστε έτοιμοι."
            return "Είμαι έτοιμη όποτε είστε κι εσείς. Παρακαλώ επαναλάβετε το πλήρες τηλέφωνο."

        if support_state == "ticket_name":
            return "Όποτε είστε έτοιμοι, παρακαλώ πείτε μου το πλήρες όνομά σας για να δημιουργήσω το αίτημα υποστήριξης."
        if support_state == "ticket_phone":
            return "Παρακαλώ πείτε μου το τηλέφωνό σας όταν είστε έτοιμοι."
        if support_state == "ticket_email":
            return "Παρακαλώ πείτε μου το email σας όταν είστε έτοιμοι."
        if support_state == "ticket_issue":
            return "Παρακαλώ περιγράψτε το πρόβλημα με μία ή δύο προτάσεις όταν είστε έτοιμοι."
        if support_state == "ticket_confirm":
            return "Παρακαλώ πείτε ναι για δημιουργία ή όχι για ακύρωση."

        if state.last_issue:
            if phase == 0:
                return "Είμαι ακόμα εδώ για να σας βοηθήσω με το θέμα σας. Παρακαλώ συνεχίστε όποτε είστε έτοιμοι."
            return "Μην βιάζεστε. Μπορώ να συνεχίσω να σας βοηθάω όποτε είστε έτοιμοι."

        if phase == 0:
            return "Είμαι εδώ όποτε είστε έτοιμοι."
        if phase == 1:
            return "Πάρτε τον χρόνο σας. Είμαι ακόμα εδώ."
        return "Μπορώ να συνεχίσω όποτε είστε έτοιμοι."

    # Simple silence monitor
    async def _silence_monitor():
        while not state.should_end:
            await asyncio.sleep(1.0)
            if not state.silence_enabled or not state.waiting_for_user or state.lookup_inflight or state.ticket_inflight:
                continue
            now = time.time()
            if now < state.silence_snooze_until:
                continue
            # Disconnect if timeout reached
            if (now - state.last_user_activity) > state.silence_timeout_s and (now - state.last_agent_activity) > state.silence_timeout_s:
                # If max_prompts is 0 or we've reached the limit, disconnect.
                if state.silence_max_prompts <= 0 or state.silence_prompt_count >= state.silence_max_prompts:
                    room_log("SILENCE_TERMINATION", count=state.silence_prompt_count)
                    state.silence_enabled = False # Stop further monitor checks
                    await agent.say("Δεν σας ακούω. Θα κλείσω την κλήση τώρα. Γεια σας!", allow_interruptions=True)
                    await asyncio.sleep(5.0) # Wait for audio to reach user
                    state.should_end = True
                    state.disconnect_reason = "silence_termination"
                    break

                text = _contextual_silence_prompt()
            state.silence_prompt_count += 1
            # Snooze for 15s to allow the agent to finish speaking and the user to react.
            state.silence_snooze_until = time.time() + 15.0
            room_log("SILENCE_PROMPT", count=state.silence_prompt_count, text=text)
            await agent.say(text, allow_interruptions=True)

    silence_task = asyncio.create_task(_silence_monitor())

    # Wait until session ends
    while not state.should_end:
        await asyncio.sleep(0.5)

    # Short delay to allow the last spoken sentence (e.g., closing message) to reach the user.
    await asyncio.sleep(6.0)

    # Cleanup and call end
    silence_task.cancel()
    transcript_text = "\n".join(conversation_transcript)
    await end_call_in_db(
        call_id=call_id,
        room_name=ctx.room.name,
        status="completed",
        duration_seconds=None,
        disconnect_reason=state.disconnect_reason,
        transcript=transcript_text or None,
    )
    room_log("CALL_END", transcript_lines=len(conversation_transcript))

    try:
        if bg_audio_player:
            await bg_audio_player.stop()
            room_log("BG_AUDIO_STOP", stopped=True)
    except Exception as e:
        room_log("BG_AUDIO_STOP_ERROR", error=str(e))

    try:
        if ctx.room and ctx.room.isconnected():
            await ctx.room.disconnect()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Worker boot
# -----------------------------------------------------------------------------


def prewarm(proc: JobProcess):
    logger.info("Elena EL prewarm complete")


def run_agent():
    log_level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    init_timeout = _as_float(os.getenv("LIVEKIT_AGENTS_INITIALIZE_TIMEOUT", "60"), 60.0, min_value=5.0, max_value=300.0)
    shutdown_timeout = _as_float(os.getenv("LIVEKIT_AGENTS_SHUTDOWN_TIMEOUT", "60"), 60.0, min_value=10.0, max_value=300.0)
    idle_procs = _as_int(os.getenv("LIVEKIT_AGENTS_NUM_IDLE_PROCESSES", "1"), 1, min_value=0, max_value=8)
    load_threshold = _as_float(os.getenv("LIVEKIT_AGENTS_LOAD_THRESHOLD", "0.9"), 0.9, min_value=0.1, max_value=1.0)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
            initialize_process_timeout=init_timeout,
            shutdown_process_timeout=shutdown_timeout,
            num_idle_processes=idle_procs,
            load_threshold=load_threshold,
        )
    )


if __name__ == "__main__":
    run_agent()
