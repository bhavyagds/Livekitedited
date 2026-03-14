"""
Greek Prosody and Voice Modulation System
Σύστημα Προσωδίας και Χροιάς Φωνής για Ελληνικό AI Voice Assistant

Controls tone, rhythm, energy and pauses based on sentence meaning.
Uses SSML (Speech Synthesis Markup Language) for ElevenLabs TTS.
"""

import re
from enum import Enum
from typing import Tuple, List
from dataclasses import dataclass


class ProsodyType(Enum):
    """Κατηγορίες Προσωδίας"""
    DECL = "declaration"      # Δήλωση
    QYES = "yes_no_question"  # Ερώτηση ναι ή όχι
    QINFO = "info_question"   # Ερώτηση πληροφορίας
    WAIT = "wait"             # Αναμονή
    CONFIRM = "confirm"       # Επιβεβαίωση
    ALERT = "alert"           # Προειδοποίηση
    EMPH = "emphasis"         # Έμφαση
    TRAIL = "trailing"        # Εκκρεμότητα


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
    # Κατηγορία: (Τόνος, Ρυθμός, Ενέργεια, Παύση πριν, Παύση μετά)
    ProsodyType.DECL: ProsodySettings(
        pitch="medium",
        rate="medium",
        volume="medium",
        break_before=0,
        break_after=300
    ),
    ProsodyType.QYES: ProsodySettings(
        pitch="high",       # ανεβαίνει στο τέλος
        rate="slow",        # ελαφρώς πιο αργός
        volume="medium",
        break_before=200,
        break_after=500     # παύση πριν το τέλος
    ),
    ProsodyType.QINFO: ProsodySettings(
        pitch="high",       # ανεβαίνει στην αρχή
        rate="medium",
        volume="medium",
        break_before=0,
        break_after=400
    ),
    ProsodyType.WAIT: ProsodySettings(
        pitch="low",        # κατεβαίνει
        rate="slow",        # αργός
        volume="soft",      # χαμηλή ενέργεια
        break_before=300,
        break_after=800     # μεγάλη παύση
    ),
    ProsodyType.CONFIRM: ProsodySettings(
        pitch="high",       # ανεβαίνει στο τέλος
        rate="medium",
        volume="medium",    # ζεστή
        break_before=0,
        break_after=300
    ),
    ProsodyType.ALERT: ProsodySettings(
        pitch="high",       # κοφτός
        rate="fast",        # γρήγορος
        volume="loud",      # υψηλή ενέργεια
        break_before=400,
        break_after=400
    ),
    ProsodyType.EMPH: ProsodySettings(
        pitch="x-high",     # αυξημένος
        rate="slow",        # αργός
        volume="x-loud",    # πολύ υψηλή
        break_before=300,
        break_after=200
    ),
    ProsodyType.TRAIL: ProsodySettings(
        pitch="low",        # πέφτει
        rate="x-slow",      # επιβραδύνεται
        volume="soft",      # χαμηλή
        break_before=0,
        break_after=1000    # μεγάλη στο τέλος
    ),
}


# =============================================================================
# DETECTION PATTERNS
# =============================================================================

# QYES: Yes/No questions - starts with these words
QYES_STARTERS = [
    r'^μπορείτε',
    r'^θέλετε',
    r'^είναι',
    r'^έχετε',
    r'^θα\s',
    r'^να\s',
    r'^μήπως',
    r'^είστε',
]

# QINFO: Information questions - starts with these words
QINFO_STARTERS = [
    r'^τι\s',
    r'^ποιος',
    r'^ποια',
    r'^ποιο',
    r'^πότε',
    r'^πού\s',
    r'^γιατί',
    r'^πώς',
    r'^πόσο',
    r'^πόση',
    r'^πόσα',
    r'^πόσες',
]

# WAIT: Waiting indicators
WAIT_KEYWORDS = [
    r'περιμένετε',
    r'ένα λεπτό',
    r'λοιπόν',
    r'μισό λεπτό',
    r'παρακαλώ περιμένετε',
]

# CONFIRM: Confirmation indicators
CONFIRM_KEYWORDS = [
    r'σωστά[;?]?$',
    r'έτσι[;?]?$',
    r'εντάξει[;?]?$',
    r'καταλάβατε[;?]?$',
    r'ok[;?]?$',
    r'οκ[;?]?$',
    r'κατάλαβα',
    r'ακριβώς',
]

# ALERT: Warning indicators
ALERT_KEYWORDS = [
    r'προσέξτε',
    r'προσοχή',
    r'\bόχι\b',
    r'\bμην\b',
    r'σταματήστε',
    r'λάθος',
    r'σημαντικό',
    r'επείγον',
]

# EMPH: Emphasis indicators
EMPH_KEYWORDS = [
    r'πολύ',
    r'τελείως',
    r'απολύτως',
    r'οπωσδήποτε',
    r'σίγουρα',
    r'αμέσως',
]


def detect_prosody_type(text: str) -> ProsodyType:
    """
    Detect the prosody category for a Greek sentence.
    
    Args:
        text: The Greek text to analyze
        
    Returns:
        ProsodyType enum value
    """
    text_lower = text.lower().strip()
    
    # Check for trailing (ends with ...)
    if text.endswith('...') or text.endswith('…'):
        return ProsodyType.TRAIL
    
    # Check for emphasis (uppercase or exclamation)
    if re.search(r'[Α-Ω]{3,}', text) or text.endswith('!'):
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
    if text.endswith('?') or text.endswith(';'):
        for pattern in QYES_STARTERS:
            if re.match(pattern, text_lower):
                return ProsodyType.QYES
    
    # Check for info question
    for pattern in QINFO_STARTERS:
        if re.match(pattern, text_lower):
            return ProsodyType.QINFO
    
    # Check if ends with question mark (general question)
    if text.endswith('?') or text.endswith(';'):
        return ProsodyType.QYES
    
    # Default to declaration
    return ProsodyType.DECL


def add_pauses(text: str) -> str:
    """
    Add SSML break tags at natural pause points.
    
    Pause points:
    - κόμμα (comma): small pause (200ms)
    - τελεία (period): medium pause (400ms)
    - και, αλλά, ή, ώστε, λοιπόν: small pause (150ms)
    """
    # Add pause after comma
    text = re.sub(r',\s*', ', <break time="200ms"/> ', text)
    
    # Add pause after period (but not at end)
    text = re.sub(r'\.\s+(?=[Α-Ωα-ωA-Za-z])', '. <break time="400ms"/> ', text)
    
    # Add pause before/after conjunctions
    conjunctions = ['και', 'αλλά', 'ή', 'ώστε', 'λοιπόν', 'επομένως', 'όμως']
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


def process_sentence(text: str) -> str:
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
    text_with_pauses = add_pauses(text)
    
    # Apply prosody wrapper
    result = apply_prosody(text_with_pauses, prosody_type)
    
    return result


def convert_to_ssml(text: str) -> str:
    """
    Convert Greek text to SSML with full prosody support.
    
    This is the main function to use. It:
    1. Splits text into sentences
    2. Detects prosody type for each
    3. Applies appropriate voice settings
    4. Adds pauses
    
    Args:
        text: The full Greek text to convert
        
    Returns:
        Complete SSML document
    """
    # Split into sentences (by . ? ! ; ...)
    sentences = re.split(r'(?<=[.?!;…])\s+', text)
    
    processed_sentences = []
    for sentence in sentences:
        if sentence.strip():
            processed = process_sentence(sentence.strip())
            processed_sentences.append(processed)
    
    # Combine into SSML document
    ssml_content = ' '.join(processed_sentences)
    
    # Wrap in speak tags
    ssml = f'<speak>{ssml_content}</speak>'
    
    return ssml


def apply_greek_prosody(text: str, use_ssml: bool = True) -> str:
    """
    Main entry point: Apply Greek prosody rules to text.
    
    Args:
        text: The Greek text to process
        use_ssml: Whether to output SSML (True) or plain text with markers (False)
        
    Returns:
        Processed text ready for TTS
    """
    if use_ssml:
        return convert_to_ssml(text)
    else:
        # For non-SSML mode, just add natural pauses with punctuation
        return add_pauses(text)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    test_sentences = [
        # QYES - Yes/No questions
        "Θέλετε να συνεχίσουμε;",
        "Είναι αυτό σωστό;",
        "Μπορείτε να περιμένετε;",
        
        # QINFO - Information questions
        "Πόσο κοστίζει;",
        "Πού βρίσκεστε;",
        "Τι θέλετε να μάθετε;",
        
        # DECL - Declarations
        "Η παραγγελία σας είναι έτοιμη.",
        "Θα σας καλέσουμε αύριο.",
        
        # WAIT - Waiting
        "Περιμένετε ένα λεπτό...",
        "Λοιπόν, ας δούμε...",
        
        # CONFIRM - Confirmation
        "Εντάξει;",
        "Σωστά;",
        
        # ALERT - Warning
        "Προσέξτε, αυτό είναι σημαντικό!",
        "Όχι, αυτό δεν είναι σωστό.",
        
        # EMPH - Emphasis
        "Είναι ΠΟΛΥ σημαντικό!",
        "Απολύτως σωστό!",
        
        # TRAIL - Trailing
        "Θα δούμε...",
        "Λοιπόν, σκέφτομαι...",
    ]
    
    print("Greek Prosody Detection Test:")
    print("=" * 60)
    
    for sentence in test_sentences:
        prosody = detect_prosody_type(sentence)
        settings = PROSODY_MAP[prosody]
        print(f"\n{sentence}")
        print(f"  Type: {prosody.name} ({prosody.value})")
        print(f"  Pitch: {settings.pitch}, Rate: {settings.rate}, Volume: {settings.volume}")
    
    print("\n" + "=" * 60)
    print("\nSSML Output Example:")
    sample = "Η παραγγελία σας είναι έτοιμη. Θέλετε να συνεχίσουμε;"
    ssml = convert_to_ssml(sample)
    print(f"\nInput: {sample}")
    print(f"\nSSML:\n{ssml}")
