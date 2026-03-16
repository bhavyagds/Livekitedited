"""
Abuse and Hostility Handling System
Σύστημα Διαχείρισης Κακοποιητικής Συμπεριφοράς

Handles abusive language by:
- Not escalating conflict
- Not responding aggressively  
- Protecting the brand
- Redirecting to productive conversation
- Setting limits without provoking

Supports both Greek and English.
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class AbuseLevel(Enum):
    """Abuse intensity levels"""
    NONE = "none"           # No abuse detected
    A1 = "rudeness"         # Αγένεια, εκνευρισμός - Rudeness, frustration
    A2 = "insults"          # Βρισιές - Insults, swearing
    A3 = "threats"          # Απειλές ή προσβολή ταυτότητας - Threats or identity attacks


@dataclass
class AbuseResponse:
    """Response configuration for abuse handling"""
    level: AbuseLevel
    response_el: str  # Greek response
    response_en: str  # English response
    should_escalate: bool = False
    should_end_call: bool = False


# =============================================================================
# ABUSE DETECTION PATTERNS
# =============================================================================

# Greek insults and abusive words (A2 level)
GREEK_INSULTS = [
    r'μαλάκ[αεοι]',          # μαλάκα, μαλάκες, μαλάκο
    r'μαλακί[αες]',          # μαλακία, μαλακίες
    r'γαμ[ωώήι]',            # γαμώ, γαμήσου
    r'πούστ[ηι]',            # πούστη, πούστι
    r'σκατ[άα]',             # σκατά
    r'άχρηστ[οηεια]',        # άχρηστος, άχρηστη, άχρηστοι
    r'ηλίθι[οεαοι]',         # ηλίθιος, ηλίθιε, ηλίθιοι
    r'βλάκ[αες]',            # βλάκας, βλάκες
    r'κόπαν[οε]',            # κόπανος, κόπανε
    r'χαζ[όοέεήοι]',         # χαζός, χαζοί
    r'απατεών[αες]?',        # απατεώνας, απατεώνες
    r'κλέφτ[ηες]',           # κλέφτης, κλέφτες
    r'ψεύτ[ηες]',            # ψεύτης, ψεύτες
    r'πουτάν[αες]',          # πουτάνα, πουτάνες
    r'καριόλ[ηα]',           # καριόλη, καριόλα
    r'μουν[ίι]',             # μουνί
    r'αρχίδ[ιαο]',           # αρχίδι, αρχίδια
]

# Greek rudeness (A1 level)
GREEK_RUDENESS = [
    r'\bντροπή\s*σου\b',
    r'\bαίσχος\b',
    r'\bαπαράδεκτ[οηε]\b',
    r'\bάθλι[οεα]\b',
    r'\bκατάπτυστ[οηε]\b',
    r'\bγελοί[οεα]\b',
    r'\bαστεί[οεα]\s+εταιρεία\b',
    r'\bδεν\s+έχετε\s+ιδέα\b',
    r'\bξέρετε\s+τι\s+κάνετε\b',
    r'\bτι\s+εταιρεία\s+είστε\b',
    r'\bμε\s+κοροϊδεύετε\b',
    r'\bμε\s+δουλεύετε\b',
]

# Greek threats (A3 level)
GREEK_THREATS = [
    r'\bθα\s+σε\s+σκοτώσω\b',
    r'\bθα\s+σας\s+καταστρέψω\b',
    r'\bθα\s+σας\s+κάνω\s+μήνυση\b',
    r'\bθα\s+σας\s+καταγγείλω\b',
    r'\bθα\s+έρθω\s+εκεί\b',
    r'\bθα\s+το\s+μετανιώσετε\b',
    r'\bπερίμενε\s+να\s+δεις\b',
    r'\bξέρω\s+πού\s+είστε\b',
    r'\bθα\s+το\s+πληρώσετε\b',
]

# English insults (A2 level)
ENGLISH_INSULTS = [
    r'\b(?:fuck|f\*ck|fck)\b',
    r'\b(?:fucking|f\*cking)\b',
    r'\b(?:shit|sh\*t)\b',
    r'\b(?:bullshit|bs)\b',
    r'\b(?:damn|dammit)\b',
    r'\b(?:ass|a\*\*)\b',
    r'\b(?:asshole|a\*\*hole)\b',
    r'\bidiot[s]?\b',
    r'\bstupid\b',
    r'\bmoron[s]?\b',
    r'\bimbecile[s]?\b',
    r'\bdumb\b',
    r'\b(?:bitch|b\*tch)\b',
    r'\bscam(?:mer)?[s]?\b',
    r'\bthief\b',
    r'\bthieves\b',
    r'\bliar[s]?\b',
    r'\bfraud[s]?\b',
    r'\bcrook[s]?\b',
    r'\buseless\b',
    r'\bpathetic\b',
    r'\bjoke\s+of\s+a\s+company\b',
]

# English rudeness (A1 level)
ENGLISH_RUDENESS = [
    r'\b(?:this|you)\s+(?:is|are)\s+(?:ridiculous|unacceptable)\b',
    r'\bshame\s+on\s+you\b',
    r'\bdisgraceful\b',
    r'\bdisgusting\b',
    r'\bunbelievable\b',
    r'\byou\s+have\s+no\s+idea\b',
    r'\bdo\s+you\s+(?:even\s+)?know\s+what\s+you\'?re\s+doing\b',
    r'\bwhat\s+kind\s+of\s+company\b',
    r'\bare\s+you\s+(?:kidding|joking)\b',
    r'\bwaste\s+of\s+(?:time|money)\b',
    r'\bi\'?m\s+(?:so\s+)?sick\s+of\b',
    r'\bi\'?ve\s+had\s+(?:it|enough)\b',
]

# English threats (A3 level)
ENGLISH_THREATS = [
    r'i\'?ll\s+(?:kill|destroy)',
    r'i\'?m\s+going\s+to\s+sue',
    r'i\'?ll\s+(?:sue|report)\s+you',
    r'i\s+will\s+sue',              # Added
    r'see\s+you\s+in\s+court',
    r'i\'?ll\s+(?:come\s+)?(?:there|over)',
    r'you\'?ll\s+(?:regret|pay\s+for)',
    r'just\s+(?:you\s+)?wait',
    r'i\s+know\s+where\s+you',
    r'i\'?ll\s+(?:find|get)\s+you',
    r'you\'?re\s+(?:going\s+to\s+)?(?:pay|dead)',
    r'gonna\s+sue',                  # Added
    r'will\s+(?:report|sue)',        # Added
]


# =============================================================================
# RESPONSES FOR EACH ABUSE LEVEL
# =============================================================================

ABUSE_RESPONSES = {
    AbuseLevel.A1: AbuseResponse(
        level=AbuseLevel.A1,
        response_el="Καταλαβαίνουμε ότι είστε εκνευρισμένοι. Θα θέλατε να μας πείτε τι ακριβώς χρειάζεστε;",
        response_en="We understand you're frustrated. Would you like to tell us exactly what you need?",
        should_escalate=False,
        should_end_call=False,
    ),
    AbuseLevel.A2: AbuseResponse(
        level=AbuseLevel.A2,
        response_el="Θα θέλαμε να σας βοηθήσουμε, όμως παρακαλούμε να κρατήσουμε τον διάλογο σε κόσμιο επίπεδο. Πείτε μας τι μπορούμε να κάνουμε για εσάς.",
        response_en="We'd like to help you, but please let's keep this conversation respectful. Tell us what we can do for you.",
        should_escalate=False,
        should_end_call=False,
    ),
    AbuseLevel.A3: AbuseResponse(
        level=AbuseLevel.A3,
        response_el="Δεν μπορούμε να συνεχίσουμε τη συνομιλία με αυτόν τον τρόπο. Αν θέλετε βοήθεια, παρακαλούμε μιλήστε μας με σεβασμό.",
        response_en="We cannot continue this conversation in this manner. If you'd like assistance, please speak to us with respect.",
        should_escalate=True,
        should_end_call=False,
    ),
}

# Final warning response (after 3 abuse incidents)
FINAL_WARNING = AbuseResponse(
    level=AbuseLevel.A3,
    response_el="Λυπούμαστε, αλλά θα πρέπει να τερματίσουμε αυτή τη συνομιλία. Μπορείτε να επικοινωνήσετε μαζί μας μέσω email στο support@meallion.com αν χρειάζεστε βοήθεια.",
    response_en="We're sorry, but we need to end this conversation. You can contact us via email at support@meallion.com if you need assistance.",
    should_escalate=True,
    should_end_call=True,
)


# =============================================================================
# ABUSE TRACKER (Per-session)
# =============================================================================

class AbuseTracker:
    """
    Tracks abuse incidents within a session.
    Escalates response after repeated offenses.
    """
    
    def __init__(self):
        self.incident_count = 0
        self.max_level_seen = AbuseLevel.NONE
        self.history: List[AbuseLevel] = []
    
    def record_incident(self, level: AbuseLevel) -> None:
        """Record an abuse incident."""
        if level != AbuseLevel.NONE:
            self.incident_count += 1
            self.history.append(level)
            if level.value > self.max_level_seen.value:
                self.max_level_seen = level
            logger.warning(f"⚠️ Abuse incident #{self.incident_count}: {level.value}")
    
    def get_response_level(self, detected_level: AbuseLevel) -> AbuseLevel:
        """
        Get the appropriate response level based on history.
        Escalates after repeated incidents.
        """
        # If this is A3 or 3+ incidents, always respond at A3 level
        if detected_level == AbuseLevel.A3 or self.incident_count >= 2:
            return AbuseLevel.A3
        
        # If this is second incident, escalate to A2
        if self.incident_count >= 1:
            return max(detected_level, AbuseLevel.A2, key=lambda x: list(AbuseLevel).index(x))
        
        # First incident, respond at detected level
        return detected_level
    
    def should_end_call(self) -> bool:
        """Check if call should be terminated."""
        return self.incident_count >= 3
    
    def reset(self) -> None:
        """Reset tracker for new session."""
        self.incident_count = 0
        self.max_level_seen = AbuseLevel.NONE
        self.history.clear()


def detect_abuse_level(text: str, language: str = "el") -> AbuseLevel:
    """
    Detect the abuse level in user input.
    
    Args:
        text: User's message
        language: 'el' for Greek, 'en' for English
        
    Returns:
        AbuseLevel enum value
    """
    text_lower = text.lower()
    
    # Check for aggressive formatting (ALL CAPS with multiple exclamation marks)
    has_aggressive_formatting = bool(re.search(r'[A-ZΑ-Ω]{5,}', text) and re.search(r'!{2,}', text))
    
    if language == "el":
        threats = GREEK_THREATS
        insults = GREEK_INSULTS
        rudeness = GREEK_RUDENESS
    else:
        threats = ENGLISH_THREATS
        insults = ENGLISH_INSULTS
        rudeness = ENGLISH_RUDENESS
    
    # Check for threats (A3)
    for pattern in threats:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.warning(f"🚨 Threat detected: {pattern}")
            return AbuseLevel.A3
    
    # Check for insults (A2)
    for pattern in insults:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.warning(f"⚠️ Insult detected: {pattern}")
            return AbuseLevel.A2
    
    # Aggressive formatting alone is A2
    if has_aggressive_formatting:
        logger.warning("⚠️ Aggressive formatting detected (CAPS + !!!)")
        return AbuseLevel.A2
    
    # Check for rudeness (A1)
    for pattern in rudeness:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.info(f"📢 Rudeness detected: {pattern}")
            return AbuseLevel.A1
    
    return AbuseLevel.NONE


def get_abuse_response(
    level: AbuseLevel,
    language: str = "el",
    tracker: Optional[AbuseTracker] = None
) -> Optional[str]:
    """
    Get the appropriate response for an abuse level.
    
    Args:
        level: Detected abuse level
        language: 'el' for Greek, 'en' for English
        tracker: Optional abuse tracker for escalation
        
    Returns:
        Response string or None if no abuse
    """
    if level == AbuseLevel.NONE:
        return None
    
    # Record incident and get escalated level if needed
    if tracker:
        tracker.record_incident(level)
        
        # Check if we need to end the call
        if tracker.should_end_call():
            logger.error("🛑 Too many abuse incidents - ending call")
            return FINAL_WARNING.response_el if language == "el" else FINAL_WARNING.response_en
        
        # Get potentially escalated response level
        level = tracker.get_response_level(level)
    
    response_config = ABUSE_RESPONSES.get(level)
    if response_config:
        return response_config.response_el if language == "el" else response_config.response_en
    
    return None


def format_abuse_response_ssml(response: str, language: str = "el") -> str:
    """
    Format abuse response with calm prosody SSML.
    
    Abuse responses use:
    - Low pitch (de-escalating)
    - Slow rate (calming)
    - Stable/soft volume
    - No emphasis
    """
    ssml = f'''<speak>
<prosody pitch="low" rate="slow" volume="soft">
<break time="500ms"/>
{response}
<break time="300ms"/>
</prosody>
</speak>'''
    return ssml


def check_and_respond_to_abuse(
    text: str,
    language: str = "el",
    tracker: Optional[AbuseTracker] = None,
    use_ssml: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Main function: Check for abuse and get formatted response.
    
    Args:
        text: User's message
        language: 'el' for Greek, 'en' for English
        tracker: Optional abuse tracker
        use_ssml: Whether to format with SSML
        
    Returns:
        Tuple of (abuse_detected: bool, response: Optional[str])
    """
    level = detect_abuse_level(text, language)
    
    if level == AbuseLevel.NONE:
        return False, None
    
    response = get_abuse_response(level, language, tracker)
    
    if response and use_ssml:
        response = format_abuse_response_ssml(response, language)
    
    return True, response


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Abuse Detection Test")
    print("=" * 60)
    
    # Greek tests
    greek_tests = [
        "Ποιά είναι η κατάσταση της παραγγελίας μου;",  # Normal
        "Αυτό είναι απαράδεκτο! Τι εταιρεία είστε;",    # A1
        "Είστε μαλάκες! Γαμώ την εταιρεία σας!",        # A2
        "Θα σας καταστρέψω! Θα το μετανιώσετε!",        # A3
    ]
    
    print("\nGreek Tests:")
    for text in greek_tests:
        level = detect_abuse_level(text, "el")
        print(f"  {level.name:6} | {text[:50]}...")
    
    # English tests
    english_tests = [
        "What's the status of my order?",               # Normal
        "This is unacceptable! What kind of company is this?",  # A1
        "You're all idiots! This is fucking ridiculous!",      # A2
        "I'll sue you! You'll regret this!",            # A3
    ]
    
    print("\nEnglish Tests:")
    for text in english_tests:
        level = detect_abuse_level(text, "en")
        print(f"  {level.name:6} | {text[:50]}...")
    
    # Test escalation
    print("\n" + "=" * 60)
    print("Escalation Test:")
    tracker = AbuseTracker()
    
    messages = [
        ("el", "Απαράδεκτοι είστε!"),      # A1
        ("el", "Μαλάκες!"),                 # A2
        ("el", "Θα το μετανιώσετε!"),       # A3
    ]
    
    for lang, msg in messages:
        abuse_detected, response = check_and_respond_to_abuse(msg, lang, tracker, use_ssml=False)
        print(f"\n  Input: {msg}")
        print(f"  Incident #{tracker.incident_count}")
        print(f"  Response: {response[:60]}..." if response else "  No abuse")
