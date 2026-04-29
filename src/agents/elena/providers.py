import os
import logging
import time
import asyncio
from livekit.plugins import openai, silero
from livekit.agents import stt
from typing import Optional, Any
from src.agents.elena import _require_setting, room_log, get_agent_language, _current_session, get_agent_setting, _as_bool, _as_float, _as_int

logger = logging.getLogger(__name__)

def create_llm():
    """Create the LLM instance based on admin settings.
    
    Supports:
    - OpenAI: gpt-4o-mini (recommended), gpt-4o, gpt-3.5-turbo
    - Groq: llama-3.3-70b-versatile (fastest!), llama-3.1-8b-instant
    
    Provider and model are configured from admin panel.
    API keys come from environment variables.
    """
    import os
    
    # Read provider from database settings only (admin-controlled)
    provider = str(_require_setting("llm_provider")).strip().lower()
    
    if provider == "groq":
        # Groq is 10x faster than OpenAI - near instant responses
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.warning("GROQ_API_KEY not set, falling back to OpenAI")
        else:
            try:
                from livekit.plugins import openai as openai_plugin
                # Read model from database settings (admin-controlled)
                groq_model = str(_require_setting("groq_model")).strip()
                logger.info(f"⚡ Using Groq LLM: {groq_model} (ultra-fast)")
                room_log("LLM_PROVIDER", provider="groq", model=groq_model)
                return openai_plugin.LLM.with_groq(
                    model=groq_model,
                    temperature=0.3,  # Lower = faster, more focused responses
                )
            except Exception as e:
                logger.warning(f"Groq init failed, falling back to OpenAI: {e}")
    
    # Default: OpenAI
    # Read model from database settings (admin-controlled)
    openai_model = str(_require_setting("openai_model")).strip()
    logger.info(f"🤖 Using OpenAI LLM: {openai_model}")
    room_log("LLM_PROVIDER", provider="openai", model=openai_model)
    return openai.LLM(
        model=openai_model,
        temperature=0.3,  # Lower = faster, more deterministic
    )


def create_tts():
    """Create TTS with automatic fallback if ElevenLabs is unavailable."""
    import json
    import urllib.error
    import urllib.request

    class FailoverTTS:
        """Wrap a primary TTS and fall back to secondary on runtime failure."""

        def __init__(self, primary, fallback, *, primary_supports_ssml: bool = False):
            self._primary = primary
            self._fallback = fallback
            self._use_fallback = False
            self._locked_provider = None  # "primary" or "fallback"
            self._audio_emitted = False
            self._primary_supports_ssml = primary_supports_ssml
            self._lock_per_call = _as_bool(
                get_agent_setting("tts_failover_lock_per_call", True),
                default=True,
            )

        def _active(self):
            if self._locked_provider == "primary":
                return self._primary
            if self._locked_provider == "fallback":
                return self._fallback
            return self._fallback if self._use_fallback else self._primary

        def current_provider_name(self) -> str:
            return "elevenlabs" if self._active() is self._primary else "openai"

        def supports_ssml(self) -> bool:
            return self._active() is self._primary and self._primary_supports_ssml

        def _lock_to(self, provider: str):
            if not self._lock_per_call:
                return
            if self._locked_provider is None:
                self._locked_provider = provider
                _current_session["tts_provider"] = "elevenlabs" if provider == "primary" else "openai"
                logger.info("TTS provider locked to %s for this call", provider)
                room_log("TTS_LOCKED", provider=provider)

        def _switch_to_fallback(self, error: Exception):
            if self._use_fallback:
                return False
            if self._locked_provider == "primary" and self._lock_per_call:
                logger.warning(
                    "Primary TTS failed after lock; keeping provider to avoid tone change: %s",
                    error,
                )
                room_log("TTS_FAILOVER_BLOCKED", reason=str(error)[:200])
                return False
            logger.warning("Primary TTS failed, switching to fallback: %s", error)
            room_log("TTS_FAILOVER", reason=str(error)[:200])
            self._use_fallback = True
            self._lock_to("fallback")
            return True

        async def _stream_with_fallback(self, text, **kwargs):
            provider = self._active()
            try:
                async for seg in provider.stream(text, **kwargs):
                    if not self._audio_emitted:
                        self._audio_emitted = True
                        if provider is self._primary:
                            self._lock_to("primary")
                        else:
                            self._lock_to("fallback")
                    yield seg
                return
            except Exception as e:
                if provider is self._fallback:
                    raise
                if self._audio_emitted and self._lock_per_call:
                    logger.warning(
                        "Primary TTS failed mid-call; keeping provider locked to avoid tone change: %s",
                        e,
                    )
                    room_log("TTS_FAILOVER_BLOCKED", reason=str(e)[:200])
                    raise
                if not self._switch_to_fallback(e):
                    raise
                async for seg in self._fallback.stream(text, **kwargs):
                    if not self._audio_emitted:
                        self._audio_emitted = True
                        self._lock_to("fallback")
                    yield seg

        def stream(self, text=None, **kwargs):
            # LiveKit may call stream() with no args and push text later.
            if text is None:
                try:
                    return self._active().stream()
                except Exception as e:
                    if self._active() is self._fallback:
                        raise
                    self._switch_to_fallback(e)
                    return self._fallback.stream()
            return self._stream_with_fallback(text, **kwargs)

        async def synthesize(self, text, **kwargs):
            provider = self._active()
            try:
                result = await provider.synthesize(text, **kwargs)
                if not self._audio_emitted:
                    self._audio_emitted = True
                    if provider is self._primary:
                        self._lock_to("primary")
                    else:
                        self._lock_to("fallback")
                return result
            except Exception as e:
                if provider is self._fallback:
                    raise
                if self._audio_emitted and self._lock_per_call:
                    logger.warning(
                        "Primary TTS failed mid-call; keeping provider locked to avoid tone change: %s",
                        e,
                    )
                    room_log("TTS_FAILOVER_BLOCKED", reason=str(e)[:200])
                    raise
                if not self._switch_to_fallback(e):
                    raise
                result = await self._fallback.synthesize(text, **kwargs)
                if not self._audio_emitted:
                    self._audio_emitted = True
                    self._lock_to("fallback")
                return result

        def __getattr__(self, name):
            return getattr(self._active(), name)

    def create_openai_tts():
        """Fallback TTS provider using OpenAI audio API."""
        voice = str(get_agent_setting("openai_tts_voice", "alloy") or "alloy")
        model = str(get_agent_setting("openai_tts_model", "tts-1") or "tts-1")
        speed = _as_float(
            get_agent_setting("openai_tts_speed", 1.0),
            1.0,
            min_value=0.25,
            max_value=4.0,
        )
        logger.warning(f"Falling back to OpenAI TTS: model={model}, voice={voice}, speed={speed}")
        room_log("TTS_PROVIDER", provider="openai", model=model, voice=voice, speed=speed)
        _current_session["tts_provider"] = "openai"
        tts = openai.TTS(model=model, voice=voice, speed=speed)
        setattr(tts, "_supports_ssml", False)
        setattr(tts, "_provider_name", "openai")
        return tts

    def elevenlabs_available() -> bool:
        """Check whether ElevenLabs key is valid for core voice endpoints."""
        if not settings.elevenlabs_api_key:
            logger.warning("ELEVENLABS_API_KEY missing; using OpenAI TTS fallback")
            return False
        try:
            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": settings.elevenlabs_api_key},
            )
            with urllib.request.urlopen(req, timeout=8):
                pass
            return True
        except urllib.error.HTTPError as e:
            logger.warning(f"ElevenLabs auth check HTTP {e.code}; using OpenAI TTS fallback")
            return False
        except Exception as e:
            logger.warning(f"ElevenLabs auth check failed: {e}; using OpenAI TTS fallback")
            return False

    def elevenlabs_voice_exists(voice_id: str) -> bool:
        """Validate the configured ElevenLabs voice id."""
        if not voice_id:
            return False
        try:
            req = urllib.request.Request(
                f"https://api.elevenlabs.io/v1/voices/{voice_id}",
                headers={"xi-api-key": settings.elevenlabs_api_key},
            )
            with urllib.request.urlopen(req, timeout=8):
                pass
            return True
        except urllib.error.HTTPError as e:
            logger.warning(f"ElevenLabs voice check failed for {voice_id}: HTTP {e.code}")
            return False
        except Exception as e:
            logger.warning(f"ElevenLabs voice check failed for {voice_id}: {e}")
            return False

    def elevenlabs_synthesis_available() -> bool:
        """Check if ElevenLabs account can still synthesize speech."""
        try:
            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/user/subscription",
                headers={"xi-api-key": settings.elevenlabs_api_key},
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))

            character_count = payload.get("character_count")
            character_limit = payload.get("character_limit")
            if isinstance(character_count, int) and isinstance(character_limit, int) and character_limit > 0:
                if character_count >= character_limit:
                    logger.warning(
                        "ElevenLabs character quota exhausted (%s/%s). Using OpenAI TTS fallback.",
                        character_count,
                        character_limit,
                    )
                    return False
            return True
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 429):
                logger.warning(f"ElevenLabs synthesis unavailable (HTTP {e.code}). Using OpenAI TTS fallback.")
                return False
            logger.warning(f"ElevenLabs subscription check HTTP {e.code}; using OpenAI TTS fallback")
            return False
        except Exception as e:
            logger.warning(f"ElevenLabs subscription check failed: {e}; using OpenAI TTS fallback")
            return False

    tts_provider = str(get_agent_setting("tts_provider", "elevenlabs") or "elevenlabs").lower()
    if tts_provider == "openai":
        return create_openai_tts()

    enable_failover = _as_bool(get_agent_setting("tts_failover_enabled", True), default=True)

    if not elevenlabs_available():
        return create_openai_tts()
    if not elevenlabs_synthesis_available():
        return create_openai_tts()

    agent_lang = get_agent_language()
    auto_language_switch = _as_bool(get_agent_setting("auto_language_switch", False), default=False)
    
    # CRITICAL: Select the correct model based on language
    # eleven_turbo_v2 is ENGLISH ONLY - Greek requires multilingual model
    configured_model = settings.elevenlabs_model
    if auto_language_switch:
        if configured_model not in {"eleven_multilingual_v2", "eleven_turbo_v2_5"}:
            tts_model = "eleven_multilingual_v2"
            logger.warning(
                "TTS: overriding %s -> %s for auto language switching",
                configured_model,
                tts_model,
            )
        else:
            tts_model = configured_model
    elif agent_lang == "el" and configured_model == "eleven_turbo_v2":
        # Override to multilingual for Greek support
        tts_model = "eleven_multilingual_v2"
        logger.warning("TTS: overriding %s -> %s for Greek support", configured_model, tts_model)
    elif agent_lang == "el" and "turbo" in configured_model.lower() and "v2_5" not in configured_model:
        # eleven_turbo_v2 doesn't support Greek, v2.5 does
        tts_model = "eleven_turbo_v2_5"
        logger.warning("TTS: overriding %s -> %s for Greek support", configured_model, tts_model)
    else:
        tts_model = configured_model
    
    logger.info("TTS model selected: %s (language: %s)", tts_model, agent_lang)
    
    voice_id = str(get_agent_setting("agent_voice_id", settings.elevenlabs_voice_id) or settings.elevenlabs_voice_id)
    if not elevenlabs_voice_exists(voice_id):
        fallback_voice_id = settings.elevenlabs_voice_id
        if fallback_voice_id != voice_id and elevenlabs_voice_exists(fallback_voice_id):
            logger.warning("Configured voice_id '%s' invalid. Falling back to '%s'.", voice_id, fallback_voice_id)
            voice_id = fallback_voice_id
        else:
            logger.warning("No valid ElevenLabs voice_id available. Falling back to OpenAI TTS.")
            return create_openai_tts()

    voice_speed = _require_float_setting(
        "agent_voice_speed",
        min_value=0.5,
        max_value=1.2,
    )
    voice_stability = _require_float_setting(
        "agent_voice_stability",
        min_value=0.0,
        max_value=1.0,
    )
    voice_similarity = _as_float(
        get_agent_setting("agent_voice_similarity", settings.elevenlabs_voice_similarity),
        settings.elevenlabs_voice_similarity,
        min_value=0.0,
        max_value=1.0,
    )

    logger.info(
        "TTS voice config: voice_id=%s speed=%.2f stability=%.2f similarity=%.2f",
        voice_id,
        voice_speed,
        voice_stability,
        voice_similarity,
    )
    room_log(
        "TTS_PROVIDER",
        provider="elevenlabs",
        model=tts_model,
        voice_id=voice_id,
        speed=voice_speed,
        stability=voice_stability,
        similarity=voice_similarity,
    )
    _current_session["tts_provider"] = "elevenlabs"

    allow_advanced = _as_bool(
        get_agent_setting("elevenlabs_allow_advanced_settings", False),
        default=False,
    )

    voice_settings = elevenlabs.VoiceSettings(
        stability=voice_stability,
        similarity_boost=voice_similarity,
        # These advanced knobs can cause ElevenLabs 400/500 on some plans/voices.
        style=0.0 if allow_advanced else None,
        speed=voice_speed if allow_advanced else None,
        use_speaker_boost=True if allow_advanced else False,
    )

    if not allow_advanced:
        logger.info("ElevenLabs advanced voice settings disabled for compatibility")

    voice = elevenlabs.Voice(
        id=voice_id,
        name="Eleni",
        category="premade",
        settings=voice_settings,
    )
    tts_use_ssml = _as_bool(get_agent_setting("tts_use_ssml", False), default=False)
    primary_tts = elevenlabs.TTS(
        voice=voice,
        model=tts_model,
        enable_ssml_parsing=tts_use_ssml,
    )
    setattr(primary_tts, "_supports_ssml", tts_use_ssml)
    setattr(primary_tts, "_provider_name", "elevenlabs")
    if enable_failover:
        return FailoverTTS(
            primary_tts,
            create_openai_tts(),
            primary_supports_ssml=tts_use_ssml,
        )
    return primary_tts


def create_stt(*, is_sip_call: bool = False):
    """Create the Speech-to-Text instance optimized for speed."""
    agent_lang = get_agent_language()
    stt_lang = get_stt_language(agent_lang)
    auto_language_switch = _as_bool(get_agent_setting("auto_language_switch", False), default=False)
    stt_auto_detect = _as_bool(
        get_agent_setting("stt_auto_detect", auto_language_switch),
        default=auto_language_switch,
    )
    sip_stt_auto_detect = _as_bool(
        get_agent_setting("sip_stt_auto_detect", False),
        default=False,
    )
    effective_stt_auto_detect = stt_auto_detect and (not is_sip_call or sip_stt_auto_detect)
    openai_stt_model = str(get_agent_setting("openai_stt_model", "whisper-1") or "whisper-1").strip()
    deepgram_stt_model = str(get_agent_setting("deepgram_stt_model", "nova-3") or "nova-3").strip()
    # Bias OpenAI STT to preserve source language (Greek/English) instead of translating.
    openai_stt_prompt = str(
        get_agent_setting(
            "openai_stt_prompt",
            (
                "Transcribe exactly what is spoken. "
                "Do not translate. "
                "Keep the original spoken language. "
                "Likely languages are Greek and English."
            ),
        )
        or ""
    ).strip()

    def _create_openai_stt(*, language: Optional[str]) -> object:
        """
        Create OpenAI STT with best-effort compatibility across plugin versions.
        Tries prompt + language hints first, then gracefully falls back.
        """
        attempts: list[dict] = []
        base = {"model": openai_stt_model}

        if language:
            attempts.append({**base, "language": language, "prompt": openai_stt_prompt})
            attempts.append({**base, "language": language})
        else:
            attempts.append({**base, "prompt": openai_stt_prompt})
            attempts.append(base.copy())

        last_error: Optional[Exception] = None
        for kwargs in attempts:
            try:
                logger.info(
                    "Creating OpenAI STT: model=%s language=%s prompt=%s",
                    kwargs.get("model"),
                    kwargs.get("language", "auto"),
                    bool(kwargs.get("prompt")),
                )
                return openai.STT(**kwargs)
            except TypeError as e:
                last_error = e
                logger.warning("OpenAI STT args not supported, retrying with fallback args: %s", e)
                continue

        if last_error:
            raise last_error
        return openai.STT(model=openai_stt_model)

    def _create_deepgram_stt(*, language: Optional[str], auto_detect: bool) -> object:
        """
        Create Deepgram STT with best-effort compatibility across plugin versions.
        For auto language switching, prefer Deepgram auto language detection.
        """
        if not USE_DEEPGRAM:
            raise RuntimeError("Deepgram plugin is not available")

        base = {
            "model": deepgram_stt_model,
            "interim_results": True,
            "punctuate": True,
            "smart_format": True,
        }
        attempts: list[dict] = []

        if auto_detect:
            # Deepgram streaming mode in this SDK does not support detect_language=True.
            # "language=multi" is the best-effort auto-language option for streaming.
            attempts.append({**base, "language": "multi"})
            attempts.append(base.copy())
        elif language:
            attempts.append({**base, "language": language})
            attempts.append(base.copy())
        else:
            attempts.append(base.copy())

        last_error: Optional[Exception] = None
        for kwargs in attempts:
            try:
                logger.info(
                    "Creating Deepgram STT: model=%s language=%s detect_language=%s",
                    kwargs.get("model"),
                    kwargs.get("language", "auto"),
                    kwargs.get("detect_language", False),
                )
                return deepgram.STT(**kwargs)
            except TypeError as e:
                last_error = e
                logger.warning("Deepgram STT args not supported, retrying with fallback args: %s", e)
                continue

        if last_error:
            raise last_error
        return deepgram.STT(model=deepgram_stt_model)

    class FailoverSTT:
        """Wrap a primary STT and fall back to secondary on runtime failure."""

        class _StreamWrapper:
            def __init__(self, parent, stream):
                self._parent = parent
                self._stream = stream

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return await self._stream.__anext__()
                except Exception as e:
                    if self._parent._active() is self._parent._fallback:
                        raise
                    self._parent._switch_to_fallback(e)
                    raise

            def __getattr__(self, name):
                return getattr(self._stream, name)

        def __init__(self, primary, fallback):
            self._primary = primary
            self._fallback = fallback
            self._use_fallback = False

        def _active(self):
            return self._fallback if self._use_fallback else self._primary

        def _switch_to_fallback(self, error: Exception):
            if self._use_fallback:
                return
            logger.warning("Primary STT failed, switching to fallback: %s", error)
            room_log("STT_FAILOVER", reason=str(error)[:200])
            self._use_fallback = True

        def stream(self, *args, **kwargs):
            provider = self._active()
            try:
                stream = provider.stream(*args, **kwargs)
                return self._StreamWrapper(self, stream)
            except Exception as e:
                if provider is self._fallback:
                    raise
                self._switch_to_fallback(e)
                stream = self._fallback.stream(*args, **kwargs)
                return self._StreamWrapper(self, stream)

        async def transcribe(self, *args, **kwargs):
            provider = self._active()
            try:
                return await provider.transcribe(*args, **kwargs)
            except Exception as e:
                if provider is self._fallback:
                    raise
                self._switch_to_fallback(e)
                return await self._fallback.transcribe(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._active(), name)
    
    provider = str(get_agent_setting("stt_provider", "") or "").strip().lower()
    if not provider:
        provider = "deepgram" if USE_DEEPGRAM else "openai"

    if auto_language_switch and not effective_stt_auto_detect:
        if is_sip_call and not sip_stt_auto_detect:
            logger.info(
                "Auto language switch enabled, but SIP STT auto detect is disabled; using fixed STT language: %s",
                stt_lang,
            )
        else:
            logger.info(
                "Auto language switch enabled, but STT auto detect is disabled; using fixed STT language: %s",
                stt_lang,
            )
    
    # Use Deepgram as primary when selected; OpenAI remains the fallback.
    if provider == "deepgram" and USE_DEEPGRAM:
        use_auto_detect = auto_language_switch and effective_stt_auto_detect
        fallback_language = None if use_auto_detect else stt_lang
        fallback = _create_openai_stt(language=fallback_language)
        try:
            primary = _create_deepgram_stt(language=stt_lang, auto_detect=use_auto_detect)
        except Exception as e:
            logger.warning("Deepgram STT init failed, falling back to OpenAI STT: %s", e)
            room_log("STT_FAILOVER", reason=f"deepgram_init_failed:{str(e)[:180]}")
            if use_auto_detect:
                logger.info("Using OpenAI STT - model: %s - language: auto", openai_stt_model)
                room_log("STT_PROVIDER", provider="openai", model=openai_stt_model, language="auto")
                return _create_openai_stt(language=None)
            logger.info("Using OpenAI STT - model: %s - language: %s", openai_stt_model, stt_lang)
            room_log("STT_PROVIDER", provider="openai", model=openai_stt_model, language=stt_lang)
            return _create_openai_stt(language=stt_lang)

        stt_language_for_log = "auto" if use_auto_detect else stt_lang
        logger.info("Using Deepgram STT (priority) - model: %s - language: %s", deepgram_stt_model, stt_language_for_log)
        room_log(
            "STT_PROVIDER",
            provider="deepgram",
            model=deepgram_stt_model,
            language=stt_language_for_log,
            auto_detect=use_auto_detect,
        )
        return FailoverSTT(primary, fallback)

    if auto_language_switch and effective_stt_auto_detect:
        try:
            logger.info("Using OpenAI STT - model: %s - language: auto", openai_stt_model)
            room_log("STT_PROVIDER", provider="openai", model=openai_stt_model, language="auto")
            return _create_openai_stt(language=None)
        except TypeError as e:
            logger.warning("OpenAI STT auto language failed (%s); falling back to %s", e, stt_lang)
    
    # Fallback to OpenAI Whisper
    if provider == "deepgram" and not USE_DEEPGRAM:
        logger.warning("Deepgram requested but not available; falling back to OpenAI Whisper")
    logger.info("Using OpenAI STT - model: %s - language: %s", openai_stt_model, stt_lang)
    room_log("STT_PROVIDER", provider="openai", model=openai_stt_model, language=stt_lang)
    return _create_openai_stt(language=stt_lang)


def create_vad():
    """Create Voice Activity Detection tuned for better transcript completeness."""
    min_speech_duration = _as_float(
        get_agent_setting("vad_min_speech_duration", 0.15),
        0.15,
        min_value=0.1,
        max_value=0.8,
    )
    min_silence_duration = _as_float(
        get_agent_setting("vad_min_silence_duration", 0.45),
        0.45,
        min_value=0.2,
        max_value=1.5,
    )

    vad_backend = str(get_agent_setting("vad_backend", "silero") or "").strip().lower()
    if vad_backend in {"energy", "rms", "simple"}:
        energy_threshold = _as_float(
            get_agent_setting("energy_vad_threshold", 0.02),
            0.012,
            min_value=0.001,
            max_value=0.2,
        )
        prefix_padding = _as_float(
            get_agent_setting("energy_vad_prefix_padding", 0.15),
            0.15,
            min_value=0.0,
            max_value=0.8,
        )
        logger.info(
            "VAD backend: energy threshold=%.4f min_speech_duration=%.2fs min_silence_duration=%.2fs prefix_padding=%.2fs",
            energy_threshold,
            min_speech_duration,
            min_silence_duration,
            prefix_padding,
        )
        room_log(
            "VAD_CONFIG",
            backend="energy",
            threshold=energy_threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding=prefix_padding,
        )
        return EnergyVAD(
            threshold=energy_threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding,
        )

    vad_sample_rate = _as_int(
        get_agent_setting("vad_sample_rate", 8000),
        8000,
        min_value=8000,
        max_value=16000,
    )
    vad_activation_threshold = _as_float(
        get_agent_setting("vad_activation_threshold", 0.72),
        0.6,
        min_value=0.1,
        max_value=0.9,
    )
    vad_force_cpu = _as_bool(
        get_agent_setting("vad_force_cpu", True),
        default=True,
    )
    logger.info(
        "VAD config: sample_rate=%sHz activation_threshold=%.2f min_speech_duration=%.2fs min_silence_duration=%.2fs",
        vad_sample_rate,
        vad_activation_threshold,
        min_speech_duration,
        min_silence_duration,
    )
    room_log(
        "VAD_CONFIG",
        backend="silero",
        sample_rate=vad_sample_rate,
        activation_threshold=vad_activation_threshold,
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
    )
    return silero.VAD.load(
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
        activation_threshold=vad_activation_threshold,
        sample_rate=vad_sample_rate,
        force_cpu=vad_force_cpu,
    )


