import type { ReactNode } from 'react';

/**
 * Formats message text with highlighting for important information
 */

// Number words to digits mapping
const numberWords: { [key: string]: string } = {
  'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
  'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
  'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
  'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
  'eighteen': '18', 'nineteen': '19', 'twenty': '20'
};

/**
 * Convert spoken number sequences to digits
 * e.g., "twelve six one nine" → "12619"
 * e.g., "plus nine two three" → "+923"
 */
function convertSpokenNumbers(text: string): string {
  // Pattern to find sequences of number words (with optional "plus" at start)
  const numberWordPattern = /\b(plus\s+)?((?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty))+)\b/gi;
  
  return text.replace(numberWordPattern, (_match, plus, numbers) => {
    const words = numbers.toLowerCase().split(/\s+/);
    const digits = words.map((word: string) => numberWords[word] || word).join('');
    return (plus ? '+' : '') + digits;
  });
}

interface HighlightPattern {
  pattern: RegExp;
  className: string;
  icon?: string;
}

const patterns: HighlightPattern[] = [
  // Order numbers - MUST have # prefix or "order" keyword (e.g., #12345, order 12345)
  {
    pattern: /#\d{4,6}\b|(?:order\s*(?:number\s*)?|παραγγελία\s*)#?\d{4,6}\b/gi,
    className: 'highlight-order',
    icon: '📦'
  },
  // Ticket IDs (e.g., TICKET-123, TKT-456)
  {
    pattern: /\b(TICKET|TKT|TASK)-[A-Z0-9]+\b/gi,
    className: 'highlight-ticket',
    icon: '🎫'
  },
  // Prices (e.g., €45.99, $30, 50€)
  {
    pattern: /[€$£]\s?\d+(?:[.,]\d{2})?|\d+(?:[.,]\d{2})?\s?[€$£]/g,
    className: 'highlight-price',
    icon: '💰'
  },
  // Full dates with month names (e.g., January 19, 2026)
  {
    pattern: /\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Ιανουαρίου|Φεβρουαρίου|Μαρτίου|Απριλίου|Μαΐου|Ιουνίου|Ιουλίου|Αυγούστου|Σεπτεμβρίου|Οκτωβρίου|Νοεμβρίου|Δεκεμβρίου)\s+\d{1,2}(?:,?\s+\d{4})?\b/gi,
    className: 'highlight-date',
    icon: '📅'
  },
  // Numeric dates (e.g., 15/12/2023, 2023-12-15)
  {
    pattern: /\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b/g,
    className: 'highlight-date',
    icon: '📅'
  },
  // Status indicators (delivered, pending, processing, being prepared, etc.)
  {
    pattern: /\b(delivered|pending|processing|cancelled|confirmed|completed|shipped|in transit|being prepared|prepared|παραδόθηκε|εκκρεμεί|ακυρώθηκε)\b/gi,
    className: 'highlight-status',
    icon: '✓'
  },
  // Email addresses
  {
    pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
    className: 'highlight-email',
    icon: '📧'
  },
  // Phone numbers - must have + prefix or specific formats (avoid matching order numbers)
  {
    pattern: /\+\d{1,4}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}\b/g,
    className: 'highlight-phone',
    icon: '📞'
  },
  // Subscription indicators
  {
    pattern: /\b(subscription|συνδρομή|weekly|monthly|bi-weekly|εβδομαδιαία|μηνιαία)\b/gi,
    className: 'highlight-subscription',
    icon: '🔄'
  },
];

export function formatMessage(text: string): ReactNode[] {
  // First convert spoken numbers to digits
  const processedText = convertSpokenNumbers(text);
  
  const elements: ReactNode[] = [];
  let lastIndex = 0;
  const matches: Array<{ start: number; end: number; text: string; className: string; icon?: string }> = [];

  // Find all matches
  patterns.forEach(({ pattern, className, icon }) => {
    let match;
    
    // Reset regex lastIndex
    pattern.lastIndex = 0;
    
    while ((match = pattern.exec(processedText)) !== null) {
      matches.push({
        start: match.index,
        end: match.index + match[0].length,
        text: match[0],
        className,
        icon
      });
    }
  });

  // Sort matches by start position
  matches.sort((a, b) => a.start - b.start);

  // Remove overlapping matches (keep first match)
  const nonOverlapping = matches.filter((match, index) => {
    if (index === 0) return true;
    const prevMatch = matches[index - 1];
    return match.start >= prevMatch.end;
  });

  // Build elements
  nonOverlapping.forEach((match, index) => {
    // Add text before match
    if (match.start > lastIndex) {
      elements.push(
        <span key={`text-${index}`}>
          {processedText.slice(lastIndex, match.start)}
        </span>
      );
    }

    // Add highlighted match
    elements.push(
      <span key={`highlight-${index}`} className={match.className}>
        {match.icon && <span className="highlight-icon">{match.icon}</span>}
        {match.text}
      </span>
    );

    lastIndex = match.end;
  });

  // Add remaining text
  if (lastIndex < processedText.length) {
    elements.push(
      <span key="text-final">{processedText.slice(lastIndex)}</span>
    );
  }

  return elements.length > 0 ? elements : [processedText];
}

/**
 * Detects if message contains structured information (like order details)
 * and returns a formatted structure
 */
export function detectStructuredData(text: string): { type: string; data: any } | null {
  // Check if it looks like order details (multiple lines with specific patterns)
  const hasOrderInfo = text.includes('Order') || text.includes('Παραγγελία');
  const hasItems = text.includes('items') || text.includes('προϊόντα');
  const hasPrice = /[€$£]\s?\d+/.test(text);
  const hasStatus = /(delivered|pending|processing|παραδόθηκε|εκκρεμεί)/i.test(text);

  if (hasOrderInfo && (hasItems || hasPrice || hasStatus)) {
    return {
      type: 'order',
      data: { text }
    };
  }

  // Check if it's a confirmation message
  const hasTicket = /TICKET|TKT|TASK/i.test(text);
  const hasConfirmation = /(created|confirmed|επιβεβαιώθηκε)/i.test(text);

  if (hasTicket && hasConfirmation) {
    return {
      type: 'confirmation',
      data: { text }
    };
  }

  return null;
}
