"""
Meallion Voice AI - Phone and Number Utils
Robust digit extraction from mixed speech transcripts (English and Greek).
"""

import re

# Mapping of spoken number words to digits
NUMBER_WORDS = {
    # English
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # Greek
    "μηδέν": "0", "μηδεν": "0", "ένα": "1", "ενα": "1", "δύο": "2", "δυο": "2",
    "τρία": "3", "τρια": "3", "τέσσερα": "4", "τεσσερα": "4", "πέντε": "5", "πεντε": "5",
    "έξι": "6", "εξι": "6", "επτά": "7", "επτα": "7", "εφτά": "7", "εφτα": "7",
    "οκτώ": "8", "οκτω": "8", "εννέα": "9", "εννεα": "9",
    # Transliterations
    "miden": "0", "ena": "1", "dyo": "2", "tria": "3", "tessera": "4",
    "pente": "5", "eksi": "6", "epta": "7", "okto": "8", "ennea": "9",
}

def extract_digits(text: str) -> str:
    """
    Extract digits from a string, converting spoken words (EN/EL) to numbers.
    Preserves the + prefix if present at the start.
    """
    if not text:
        return ""
        
    has_plus = text.strip().startswith("+")
    
    # Split into words/tokens
    tokens = re.findall(r"[a-zA-Z\u0370-\u03FF0-9]+", text.lower())
    digits = []
    
    for token in tokens:
        if token in NUMBER_WORDS:
            digits.append(NUMBER_WORDS[token])
        elif token.isdigit():
            digits.append(token)
        else:
            # Check for embedded digits (e.g. "order123")
            embedded = re.sub(r"\D", "", token)
            if embedded:
                digits.append(embedded)
                
    result = "".join(digits)
    if has_plus and not result.startswith("+"):
        result = "+" + result
        
    return result
