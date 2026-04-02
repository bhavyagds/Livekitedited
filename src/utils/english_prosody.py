"""
English Prosody and Voice Modulation System
Controls tone, rhythm, energy and pauses based on sentence meaning.
Uses SSML (Speech Synthesis Markup Language) for ElevenLabs TTS.
"""

import re
from enum import Enum
from typing import Tuple, List
from dataclasses import dataclass


class ProsodyType(Enum):
    """Prosody Categories"""
    DECL = "declaration"      # Statement
    QYES = "yes_no_question"  # Yes/No question
    QINFO = "info_question"   # Information question
    WAIT = "wait"             # Waiting/Hold
    CONFIRM = "confirm"       # Confirmation
    ALERT = "alert"           # Warning/Alert
    EMPH = "emphasis"         # Emphasis
    TRAIL = "trailing"        # Trailing off


@dataclass
class ProsodySettings:
    """Voice settings for each prosody type"""
    pitch: str      # low, medium, high, x-high
    rate: str       # x-slow, slow, medium, fast, x-fast
    volume: str     # soft, medium, loud, x-loud
    break_before: int   # pause in ms before
    break_after: int    # pause in ms after


# =============================================================================
# PROSODY MAPPINGS
# =============================================================================

PROSODY_MAP = {
    ProsodyType.DECL: ProsodySettings(
        pitch="medium",
        rate="medium",
        volume="medium",
        break_before=0,
        break_after=300
    ),
    ProsodyType.QYES: ProsodySettings(
        pitch="high",       # Rising at end
        rate="slow",        # Slightly slower
        volume="medium",
        break_before=200,
        break_after=500
    ),
    ProsodyType.QINFO: ProsodySettings(
        pitch="high",       # Rising at start
        rate="medium",
        volume="medium",
        break_before=0,
        break_after=400
    ),
    ProsodyType.WAIT: ProsodySettings(
        pitch="low",        # Falling
        rate="slow",        # Slow
        volume="soft",      # Low energy
        break_before=300,
        break_after=800
    ),
    ProsodyType.CONFIRM: ProsodySettings(
        pitch="high",       # Rising at end
        rate="medium",
        volume="medium",    # Warm
        break_before=0,
        break_after=300
    ),
    ProsodyType.ALERT: ProsodySettings(
        pitch="high",       # Sharp
        rate="fast",        # Quick
        volume="loud",      # High energy
        break_before=400,
        break_after=400
    ),
    ProsodyType.EMPH: ProsodySettings(
        pitch="x-high",     # Elevated
        rate="slow",        # Slow
        volume="x-loud",    # Very high
        break_before=300,
        break_after=200
    ),
    ProsodyType.TRAIL: ProsodySettings(
        pitch="low",        # Falling
        rate="x-slow",      # Slowing down
        volume="soft",      # Low
        break_before=0,
        break_after=1000
    ),
}


# =============================================================================
# DETECTION PATTERNS
# =============================================================================

# QYES: Yes/No questions - starts with these words
QYES_STARTERS = [
    r'^can\s',
    r'^could\s',
    r'^would\s',
    r'^will\s',
    r'^do\s',
    r'^does\s',
    r'^did\s',
    r'^is\s',
    r'^are\s',
    r'^was\s',
    r'^were\s',
    r'^have\s',
    r'^has\s',
    r'^had\s',
    r'^may\s',
    r'^might\s',
    r'^shall\s',
    r'^should\s',
]

# QINFO: Information questions - starts with these words
QINFO_STARTERS = [
    r'^what\s',
    r'^who\s',
    r'^whom\s',
    r'^whose\s',
    r'^when\s',
    r'^where\s',
    r'^why\s',
    r'^how\s',
    r'^which\s',
]

# WAIT: Waiting indicators
WAIT_KEYWORDS = [
    r'please wait',
    r'one moment',
    r'just a moment',
    r'hold on',
    r'let me check',
    r'let me see',
    r'give me a second',
    r'bear with me',
]

# CONFIRM: Confirmation indicators
CONFIRM_KEYWORDS = [
    r'right\??$',
    r'correct\??$',
    r'okay\??$',
    r'ok\??$',
    r'understood\??$',
    r'is that right',
    r'is that correct',
    r'got it',
    r'exactly',
    r'that\'s right',
]

# ALERT: Warning indicators
ALERT_KEYWORDS = [
    r'\bwarning\b',
    r'\bplease note\b',
    r'\bimportant\b',
    r'\burgent\b',
    r'\bstop\b',
    r'\bdon\'t\b',
    r'\bdo not\b',
    r'\bcannot\b',
    r'\bcan\'t\b',
    r'\bwrong\b',
    r'\berror\b',
    r'\bmistake\b',
]

# EMPH: Emphasis indicators
EMPH_KEYWORDS = [
    r'\bvery\b',
    r'\bextremely\b',
    r'\babsolutely\b',
    r'\bdefinitely\b',
    r'\bcertainly\b',
    r'\bimmediately\b',
    r'\bcompletely\b',
    r'\btotally\b',
]


def detect_prosody_type(text: str) -> ProsodyType:
    """
    Detect the prosody category for an English sentence.
    
    Args:
        text: The English text to analyze
        
    Returns:
        ProsodyType enum value
    """
    text_lower = text.lower().strip()
    
    # Check for trailing (ends with ...)
    if text.endswith('...') or text.endswith('…'):
        return ProsodyType.TRAIL
    
    # Check for emphasis (uppercase words or exclamation)
    if re.search(r'\b[A-Z]{3,}\b', text) or text.endswith('!'):
        return ProsodyType.EMPH
    
    # Check for emphasis keywords
    for pattern in EMPH_KEYWORDS:
        if re.search(pattern, text_lower):
            return ProsodyType.EMPH
    
    # Check for alert
    for pattern in ALERT_KEYWORDS:
        if re.search(pattern, text_lower):
            return ProsodyType.ALERT
    
    # Check for wait
    for pattern in WAIT_KEYWORDS:
        if re.search(pattern, text_lower):
            return ProsodyType.WAIT
    
    # Check for confirmation
    for pattern in CONFIRM_KEYWORDS:
        if re.search(pattern, text_lower):
            return ProsodyType.CONFIRM
    
    # Check for yes/no question
    if text.endswith('?'):
        for pattern in QYES_STARTERS:
            if re.match(pattern, text_lower):
                return ProsodyType.QYES
    
    # Check for info question
    for pattern in QINFO_STARTERS:
        if re.match(pattern, text_lower):
            return ProsodyType.QINFO
    
    # Check if ends with question mark (general question)
    if text.endswith('?'):
        return ProsodyType.QYES
    
    # Default to declaration
    return ProsodyType.DECL


def add_pauses(text: str, use_ssml: bool = True) -> str:
    """
    Add SSML break tags at natural pause points.
    
    Pause points:
    - comma: small pause (200ms)
    - period: medium pause (400ms)
    - and, but, or, so, however: small pause (150ms)
    """
    if not use_ssml:
        # No SSML tags when SSML parsing is disabled.
        return text

    # Add pause after comma (remove the comma itself)
    text = re.sub(r',\s*', '<break time="200ms"/> ', text)

    # Add pause after period (but not at end) - remove the period
    text = re.sub(r'\.\s+(?=[A-Za-z])', '<break time="400ms"/> ', text)

    # Add pause before/after conjunctions
    conjunctions = ['and', 'but', 'or', 'so', 'however', 'therefore', 'moreover']
    for conj in conjunctions:
        pattern = rf'\s+{conj}\s+'
        replacement = f' <break time="150ms"/> {conj} <break time="150ms"/> '
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def apply_prosody(text: str, prosody_type: ProsodyType) -> str:
    """
    Wrap text in SSML prosody tags based on the detected type.
    
    Args:
        text: The text to wrap
        prosody_type: The detected prosody category
        
    Returns:
        Text wrapped in SSML tags
    """
    settings = PROSODY_MAP[prosody_type]
    
    result = ""
    
    # Add break before if needed
    if settings.break_before > 0:
        result += f'<break time="{settings.break_before}ms"/>'
    
    # Wrap in prosody tag
    result += f'<prosody pitch="{settings.pitch}" rate="{settings.rate}" volume="{settings.volume}">'
    result += text
    result += '</prosody>'
    
    # Add break after if needed
    if settings.break_after > 0:
        result += f'<break time="{settings.break_after}ms"/>'
    
    return result


def process_sentence(text: str, use_ssml: bool = True) -> str:
    """
    Process a single sentence with prosody and pauses.
    
    Args:
        text: The sentence to process
        
    Returns:
        SSML-formatted text
    """
    # Detect prosody type
    prosody_type = detect_prosody_type(text)
    
    # Add internal pauses
    text_with_pauses = add_pauses(text, use_ssml=use_ssml)
    
    # Apply prosody wrapper
    result = apply_prosody(text_with_pauses, prosody_type)
    
    return result


def convert_to_ssml(text: str) -> str:
    """
    Convert English text to SSML with full prosody support.
    
    This is the main function to use. It:
    1. Splits text into sentences
    2. Detects prosody type for each
    3. Applies appropriate voice settings
    4. Adds pauses
    
    Args:
        text: The full English text to convert
        
    Returns:
        Complete SSML document
    """
    # Split into sentences (by . ? ! ...)
    sentences = re.split(r'(?<=[.?!…])\s+', text)
    
    processed_sentences = []
    for sentence in sentences:
        if sentence.strip():
            processed = process_sentence(sentence.strip(), use_ssml=True)
            processed_sentences.append(processed)
    
    # Combine into SSML document
    ssml_content = ' '.join(processed_sentences)
    
    # Wrap in speak tags
    ssml = f'<speak>{ssml_content}</speak>'
    
    return ssml


def apply_english_prosody(text: str, use_ssml: bool = True) -> str:
    """
    Main entry point: Apply English prosody rules to text.
    
    Args:
        text: The English text to process
        use_ssml: Whether to output SSML (True) or plain text with markers (False)
        
    Returns:
        Processed text ready for TTS
    """
    if use_ssml:
        return convert_to_ssml(text)
    # No SSML parsing: return plain text (no SSML tags).
    return add_pauses(text, use_ssml=False)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    test_sentences = [
        # QYES - Yes/No questions
        "Would you like to continue?",
        "Is this correct?",
        "Can I help you with anything else?",
        
        # QINFO - Information questions
        "How much does it cost?",
        "Where are you located?",
        "What would you like to know?",
        
        # DECL - Declarations
        "Your order is ready.",
        "We will call you tomorrow.",
        
        # WAIT - Waiting
        "Please wait one moment...",
        "Let me check that for you...",
        
        # CONFIRM - Confirmation
        "Is that correct?",
        "Okay?",
        
        # ALERT - Warning
        "Please note, this is important!",
        "Warning: this action cannot be undone.",
        
        # EMPH - Emphasis
        "This is VERY important!",
        "Absolutely correct!",
        
        # TRAIL - Trailing
        "Well, let me see...",
        "I'm thinking...",
    ]
    
    print("English Prosody Detection Test:")
    print("=" * 60)
    
    for sentence in test_sentences:
        prosody = detect_prosody_type(sentence)
        settings = PROSODY_MAP[prosody]
        print(f"\n{sentence}")
        print(f"  Type: {prosody.name} ({prosody.value})")
        print(f"  Pitch: {settings.pitch}, Rate: {settings.rate}, Volume: {settings.volume}")
    
    print("\n" + "=" * 60)
    print("\nSSML Output Example:")
    sample = "Your order is ready. Would you like more details?"
    ssml = convert_to_ssml(sample)
    print(f"\nInput: {sample}")
    print(f"\nSSML:\n{ssml}")
