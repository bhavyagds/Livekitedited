"""
Meallion Voice AI - Voice Formatting Utils
Handles natural language formatting for dates, currencies, and numbers to improve TTS quality.
"""

import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Month names for natural reading
MONTHS_EN = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

MONTHS_EL = [
    "", "Ιανουαρίου", "Φεβρουαρίου", "Μαρτίου", "Απριλίου", "Μαΐου", "Ιουνίου",
    "Ιουλίου", "Αυγούστου", "Σεπτεμβρίου", "Οκτωβρίου", "Νοεμβρίου", "Δεκεμβρίου"
]

def format_date_for_voice(date_str: str, lang: str = "en") -> str:
    """
    Format a date string (YYYY/MM/DD, DD/MM/YYYY, or similar) for natural speech.
    Handles multiple date formats cleanly to avoid value errors and raw digit reading.
    
    Examples:
    - 2026/03/12, en -> "March 12th"
    - 12-03-2026, en -> "March 12th"
    - 2026/03/12, el -> "12 Μαρτίου"
    """
    if not date_str:
        return ""
        
    date_str = date_str.strip()
    
    # Strip time if ISO format like "2026-05-15T12:00:00Z"
    if "T" in date_str:
        try:
            date_str = date_str.split("T")[0]
        except Exception:
            pass

    # Normalize separators
    clean_date = date_str.replace("/", "-").strip()
    
    dt = None
    formats = [
        "%Y-%m-%d",          # YYYY-MM-DD
        "%d-%m-%Y",          # DD-MM-YYYY
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(clean_date, fmt)
            break
        except ValueError:
            # Try parsing just the prefix
            try:
                dt = datetime.strptime(clean_date[:10], fmt)
                break
            except ValueError:
                continue

    if dt:
        day = dt.day
        month = dt.month
        
        if lang == "el":
            return f"{day} {MONTHS_EL[month]}"
        else:
            # Add ordinal suffix for English
            if 11 <= day <= 13:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            return f"{MONTHS_EN[month]} {day}{suffix}"
            
    # Heuristic regex match if standard parsing fails
    # e.g., "15-5-2026" or "5-15-2026"
    match = re.search(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', date_str)
    if match:
        p1, p2, year = match.groups()
        p1_val, p2_val = int(p1), int(p2)
        # Check if p1 is day and p2 is month (European/Greek standard)
        if 1 <= p1_val <= 31 and 1 <= p2_val <= 12:
            day, month = p1_val, p2_val
        elif 1 <= p2_val <= 31 and 1 <= p1_val <= 12: # US format
            day, month = p2_val, p1_val
        else:
            day, month = None, None
            
        if day and month:
            if lang == "el":
                return f"{day} {MONTHS_EL[month]}"
            else:
                if 11 <= day <= 13:
                    suffix = "th"
                else:
                    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
                return f"{MONTHS_EN[month]} {day}{suffix}"

    # Regex heuristic for YYYY-MM-DD or YYYY/MM/DD
    match_y = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', date_str)
    if match_y:
        year, month, day = match_y.groups()
        day_val, month_val = int(day), int(month)
        if 1 <= day_val <= 31 and 1 <= month_val <= 12:
            if lang == "el":
                return f"{day_val} {MONTHS_EL[month_val]}"
            else:
                if 11 <= day_val <= 13:
                    suffix = "th"
                else:
                    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day_val % 10, "th")
                return f"{MONTHS_EN[month_val]} {day_val}{suffix}"

    logger.warning(f"Failed to format date for voice: {date_str}")
    return date_str


def clean_text_for_tts(text: str, lang: str = "en") -> str:
    """
    Clean text of symbols and punctuation that cause ElevenLabs / TTS providers 
    to make unnatural, awkward pauses.
    """
    if not text:
        return ""
        
    # Remove parentheses/brackets but keep their content, surrounded by natural pauses (commas)
    text = re.sub(r'\((.*?)\)', r', \1,', text)
    text = re.sub(r'\[(.*?)\]', r', \1,', text)
    
    # Replace colons with commas (colons create long hard pauses, commas are smooth and natural)
    text = text.replace(":", ",")
    
    # Replace semicolons with commas
    text = text.replace(";", ",")
    
    # Replace hyphens/dashes with a space or comma to prevent spelling out "dash" or hard pausing
    # e.g., list markers or stand-alone hyphens should be cleaned
    text = re.sub(r'\s+-\s+', ', ', text)
    
    # Replace generic hash symbols with "number" or remove them
    if lang == "el":
        text = text.replace("#", "νούμερο ")
    else:
        text = text.replace("#", "number ")
        
    # Normalize multiple commas/spaces/periods
    text = re.sub(r',+', ',', text)
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up double commas, comma-periods, and trailing spacing
    text = text.replace(",.", ".").replace(".,", ".").replace(", ,", ",")
    
    # Remove any extra leading/trailing commas left from replacement
    text = text.strip().strip(",")
    return text.strip()

def format_currency_for_voice(amount: float, currency: str = "EUR", lang: str = "en") -> str:
    """
    Format a currency amount for natural speech.
    
    Examples:
    - 57.50, EUR, en -> "57 euros and 50 cents"
    - 57.50, EUR, el -> "57 ευρώ και 50 λεπτά"
    """
    try:
        euros = int(amount)
        cents = int(round((amount - euros) * 100))
        
        if lang == "el":
            from src.utils.greek_numbers import format_price_greek
            return format_price_greek(amount, currency)
        else:
            currency_word = "euros" if currency in ("EUR", "€") else currency
            cents_word = "cents"
            
            if euros == 1:
                currency_word = "euro"
            if cents == 1:
                cents_word = "cent"
                
            if cents == 0:
                return f"{euros} {currency_word}"
            else:
                return f"{euros} {currency_word} and {cents} {cents_word}"
                
    except Exception as e:
        logger.warning(f"Failed to format currency for voice: {amount} {currency} ({e})")
        return f"{amount} {currency}"

def format_order_number_for_voice(order_number: str, lang: str = "en") -> str:
    """Format order number for natural speech."""
    clean_number = order_number.replace("#", "").strip()
    
    if lang == "el":
        from src.utils.greek_numbers import format_order_number_greek
        return format_order_number_greek(clean_number)
    else:
        # For English, we usually speak order numbers one digit at a time or as a whole number.
        # However, for 5-digit numbers, "one two eight seven five" is often better than "twelve thousand..."
        # If it's a long number, space it out.
        if len(clean_number) >= 4:
            return " ".join(clean_number)
        return clean_number
