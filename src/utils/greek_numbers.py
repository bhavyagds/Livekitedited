"""
Greek Number to Words Converter
Converts numeric digits to Greek word pronunciation.

Based on Greek language rules:
- Numbers 10-19 are atomic (special words)
- Special handling for 100 vs 100+n
- Thousands: χίλια (1000) vs χιλιάδες (2000+)
- Millions: εκατομμύριο (1M) vs εκατομμύρια (2M+)
"""

import re
from typing import Optional

# =============================================================================
# GREEK NUMBER DICTIONARY
# =============================================================================

# Single digits (0-9)
MONOPSIIFIA = {
    0: "μηδέν",
    1: "ένα",
    2: "δύο",
    3: "τρία",
    4: "τέσσερα",
    5: "πέντε",
    6: "έξι",
    7: "επτά",
    8: "οκτώ",
    9: "εννέα",
}

# Numbers 10-19 (atomic - don't split)
DEKA_ENNEEA = {
    10: "δέκα",
    11: "έντεκα",
    12: "δώδεκα",
    13: "δεκατρία",
    14: "δεκατέσσερα",
    15: "δεκαπέντε",
    16: "δεκαέξι",
    17: "δεκαεπτά",
    18: "δεκαοκτώ",
    19: "δεκαεννέα",
}

# Tens (20, 30, 40, ...)
DEKADES = {
    20: "είκοσι",
    30: "τριάντα",
    40: "σαράντα",
    50: "πενήντα",
    60: "εξήντα",
    70: "εβδομήντα",
    80: "ογδόντα",
    90: "ενενήντα",
}

# Hundreds (100, 200, 300, ...)
EKATONTADES = {
    100: "εκατό",      # When alone
    200: "διακόσια",
    300: "τριακόσια",
    400: "τετρακόσια",
    500: "πεντακόσια",
    600: "εξακόσια",
    700: "επτακόσια",
    800: "οκτακόσια",
    900: "εννιακόσια",
}


def number_to_greek(n: int) -> str:
    """
    Convert an integer to Greek words.
    
    Examples:
        42 -> "σαράντα δύο"
        101 -> "εκατόν ένα"
        1234 -> "χίλια διακόσια τριάντα τέσσερα"
        12619 -> "δώδεκα χιλιάδες εξακόσια δεκαεννέα"
    """
    if n < 0:
        return "μείον " + number_to_greek(-n)
    
    if n == 0:
        return MONOPSIIFIA[0]
    
    # Numbers 1-9
    if n < 10:
        return MONOPSIIFIA[n]
    
    # Numbers 10-19 (atomic)
    if n < 20:
        return DEKA_ENNEEA[n]
    
    # Numbers 20-99
    if n < 100:
        tens = (n // 10) * 10
        units = n % 10
        if units == 0:
            return DEKADES[tens]
        else:
            return f"{DEKADES[tens]} {MONOPSIIFIA[units]}"
    
    # Numbers 100-999
    if n < 1000:
        hundreds = (n // 100) * 100
        remainder = n % 100
        
        if remainder == 0:
            return EKATONTADES[hundreds]
        else:
            # Special rule: 100 becomes "εκατόν" when followed by something
            if hundreds == 100:
                return f"εκατόν {number_to_greek(remainder)}"
            else:
                return f"{EKATONTADES[hundreds]} {number_to_greek(remainder)}"
    
    # Numbers 1000-999999
    if n < 1_000_000:
        thousands = n // 1000
        remainder = n % 1000
        
        if thousands == 1:
            thousands_word = "χίλια"
        else:
            thousands_word = f"{number_to_greek(thousands)} χιλιάδες"
        
        if remainder == 0:
            return thousands_word
        else:
            return f"{thousands_word} {number_to_greek(remainder)}"
    
    # Numbers 1,000,000+
    if n < 1_000_000_000_000:
        millions = n // 1_000_000
        remainder = n % 1_000_000
        
        if millions == 1:
            millions_word = "ένα εκατομμύριο"
        else:
            millions_word = f"{number_to_greek(millions)} εκατομμύρια"
        
        if remainder == 0:
            return millions_word
        else:
            return f"{millions_word} {number_to_greek(remainder)}"
    
    # For very large numbers, just return the digits
    return str(n)


def convert_numbers_in_text(text: str) -> str:
    """
    Convert all standalone numbers in text to Greek words.
    
    Handles:
    - Order numbers like #12619 -> "δώδεκα χιλιάδες εξακόσια δεκαεννέα"
    - Prices like 58.37 -> "πενήντα οκτώ κόμμα τριάντα επτά"
    - Plain numbers like 42 -> "σαράντα δύο"
    
    Preserves context markers like # for order numbers.
    """
    
    def replace_decimal(match):
        """Replace decimal numbers like 58.37"""
        whole = match.group(1)
        decimal = match.group(2)
        
        whole_greek = number_to_greek(int(whole))
        
        # Convert decimal part digit by digit or as number
        if len(decimal) <= 2:
            decimal_greek = number_to_greek(int(decimal))
        else:
            # For longer decimals, read digit by digit
            decimal_greek = " ".join(MONOPSIIFIA[int(d)] for d in decimal)
        
        return f"{whole_greek} κόμμα {decimal_greek}"
    
    def replace_order_number(match):
        """Replace order numbers like #12619"""
        prefix = match.group(1) or ""  # # or empty
        number = match.group(2)
        greek = number_to_greek(int(number))
        # Keep the context but add Greek pronunciation
        return f"{prefix}{number}, {greek}"
    
    def replace_plain_number(match):
        """Replace standalone numbers"""
        number = match.group(0)
        # Don't convert years (4 digits starting with 19 or 20)
        if len(number) == 4 and (number.startswith("19") or number.startswith("20")):
            return number
        return number_to_greek(int(number))
    
    result = text
    
    # First, handle decimal numbers (prices)
    result = re.sub(r'(\d+)[.,](\d+)', replace_decimal, result)
    
    # Handle order numbers with # prefix
    result = re.sub(r'(#)?(\d{4,6})\b', replace_order_number, result)
    
    # Handle remaining standalone numbers (2+ digits)
    result = re.sub(r'\b(\d{2,})\b', replace_plain_number, result)
    
    return result


def format_price_greek(amount: float, currency: str = "EUR") -> str:
    """
    Format a price in Greek.
    
    Examples:
        58.37, "EUR" -> "πενήντα οκτώ ευρώ και τριάντα επτά λεπτά"
        100.00, "EUR" -> "εκατό ευρώ"
    """
    euros = int(amount)
    cents = int(round((amount - euros) * 100))
    
    currency_word = "ευρώ" if currency in ("EUR", "€") else currency
    
    euros_greek = number_to_greek(euros)
    
    if cents == 0:
        return f"{euros_greek} {currency_word}"
    else:
        cents_greek = number_to_greek(cents)
        return f"{euros_greek} {currency_word} και {cents_greek} λεπτά"


def format_order_number_greek(order_number: str) -> str:
    """
    Format an order number for Greek speech.
    
    Examples:
        "12619" -> "δώδεκα χιλιάδες εξακόσια δεκαεννέα"
        "#12619" -> "δώδεκα χιλιάδες εξακόσια δεκαεννέα"
    """
    # Remove # prefix if present
    clean_number = order_number.replace("#", "").strip()
    try:
        return number_to_greek(int(clean_number))
    except ValueError:
        return order_number


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    # Test cases
    test_numbers = [
        0, 1, 5, 9,
        10, 11, 15, 19,
        20, 21, 42, 58, 99,
        100, 101, 115, 187,
        200, 342, 619,
        1000, 1001, 1234,
        2000, 2345, 5000,
        12619, 12617,
        1000000, 2345123,
    ]
    
    print("Greek Number Conversion Test:")
    print("=" * 50)
    for n in test_numbers:
        greek = number_to_greek(n)
        print(f"{n:>12} = {greek}")
    
    print("\n" + "=" * 50)
    print("Price Formatting Test:")
    prices = [58.37, 100.00, 54.19, 0.99, 1234.56]
    for p in prices:
        print(f"€{p:.2f} = {format_price_greek(p)}")
    
    print("\n" + "=" * 50)
    print("Text Conversion Test:")
    test_texts = [
        "Order #12619 is ready",
        "Total: 58.37 EUR",
        "You have 3 items",
    ]
    for t in test_texts:
        print(f"Original: {t}")
        print(f"Greek:    {convert_numbers_in_text(t)}")
        print()
