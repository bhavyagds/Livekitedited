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
    Format a date string (YYYY/MM/DD or YYYY-MM-DD) for natural speech.
    
    Examples:
    - 2026/03/12, en -> "March 12th"
    - 2026/03/12, el -> "12 Μαρτίου"
    """
    if not date_str:
        return ""
        
    try:
        # Normalize separators
        clean_date = date_str.replace("/", "-").strip()
        # Parse YYYY-MM-DD
        dt = datetime.strptime(clean_date[:10], "%Y-%m-%d")
        
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
            
    except Exception as e:
        logger.warning(f"Failed to format date for voice: {date_str} ({e})")
        return date_str

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
