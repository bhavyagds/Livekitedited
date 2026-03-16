"""
Meallion Voice AI - Utility Functions
"""

from .greek_numbers import (
    number_to_greek,
    format_price_greek,
    format_order_number_greek,
    convert_numbers_in_text,
)

from .greek_prosody import (
    ProsodyType,
    detect_prosody_type as detect_greek_prosody_type,
    apply_greek_prosody,
    convert_to_ssml as convert_to_ssml_greek,
)

from .english_prosody import (
    detect_prosody_type as detect_english_prosody_type,
    apply_english_prosody,
    convert_to_ssml as convert_to_ssml_english,
)

from .abuse_handler import (
    AbuseLevel,
    AbuseTracker,
    detect_abuse_level,
    get_abuse_response,
    check_and_respond_to_abuse,
    format_abuse_response_ssml,
)


def apply_prosody(text: str, language: str = "el", use_ssml: bool = True) -> str:
    """
    Apply prosody rules based on language.
    
    Args:
        text: The text to process
        language: 'el' for Greek, 'en' for English
        use_ssml: Whether to output SSML
        
    Returns:
        Processed text ready for TTS
    """
    if language == "el":
        return apply_greek_prosody(text, use_ssml)
    else:
        return apply_english_prosody(text, use_ssml)


__all__ = [
    # Greek numbers
    "number_to_greek",
    "format_price_greek", 
    "format_order_number_greek",
    "convert_numbers_in_text",
    # Prosody
    "ProsodyType",
    "apply_greek_prosody",
    "apply_english_prosody",
    "apply_prosody",
    "detect_greek_prosody_type",
    "detect_english_prosody_type",
    # Abuse handling
    "AbuseLevel",
    "AbuseTracker",
    "detect_abuse_level",
    "get_abuse_response",
    "check_and_respond_to_abuse",
    "format_abuse_response_ssml",
]
