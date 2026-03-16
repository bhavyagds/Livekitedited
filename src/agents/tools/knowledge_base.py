"""
Meallion Voice AI - Knowledge Base Tool
Handles FAQ lookups and general information queries.
Loads from database (kb_items table) with file fallback.
"""

import json
import logging
from pathlib import Path
from typing import Annotated, Optional, List
import asyncio

from livekit.agents import llm

logger = logging.getLogger(__name__)


class KnowledgeBaseTool:
    """
    Knowledge base tool for Elena voice agent.
    
    Features:
    - FAQ matching by keywords
    - Brand information
    - Greeting and closing phrases
    - Database-backed FAQ items
    """

    def __init__(self):
        self.kb_data: Optional[dict] = None
        self.db_items: List[dict] = []
        self._load_knowledge_base()

    def _load_knowledge_base(self) -> None:
        """Load knowledge base from JSON file (DB items loaded on demand)."""
        kb_path = Path(__file__).parent.parent.parent.parent / "knowledge" / "meallion_faq.json"
        
        try:
            if kb_path.exists():
                with open(kb_path, "r", encoding="utf-8") as f:
                    self.kb_data = json.load(f)
                logger.info(f"Loaded knowledge base from {kb_path}")
            else:
                logger.warning(f"Knowledge base file not found: {kb_path}")
                self.kb_data = self._get_default_kb()
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            self.kb_data = self._get_default_kb()

    async def load_db_items(self) -> None:
        """Load FAQ items from database."""
        try:
            from src.services.database import DatabaseService
            db = DatabaseService()
            self.db_items = await db.get_kb_items(active_only=True)
            logger.info(f"Loaded {len(self.db_items)} FAQ items from database")
        except Exception as e:
            logger.warning(f"Could not load KB items from database: {e}")
            self.db_items = []

    async def load_db_content(self, language: str = "el") -> Optional[str]:
        """Load KB content (text) from database."""
        try:
            from src.services.database import DatabaseService
            db = DatabaseService()
            content = await db.get_kb_content(language)
            if content:
                return content.get("content", "")
        except Exception as e:
            logger.warning(f"Could not load KB content from database: {e}")
        return None

    def search_db_items(self, query: str, language: str = "el") -> Optional[str]:
        """Search FAQ items loaded from database."""
        if not self.db_items:
            return None
        
        query_lower = query.lower()
        
        # First try exact keyword match
        for item in self.db_items:
            if item.get("language", "el") != language and language != "all":
                continue
            keywords = item.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    logger.info(f"DB KB match (keyword): {keyword}")
                    return item["answer"]
        
        # Then try question text match
        for item in self.db_items:
            if item.get("language", "el") != language and language != "all":
                continue
            question_words = item["question"].lower().split()
            matches = sum(1 for word in question_words if word in query_lower)
            if matches >= 2 or any(word in query_lower for word in question_words if len(word) > 4):
                logger.info(f"DB KB match (question): {item['question'][:50]}...")
                return item["answer"]
        
        return None

    def _get_default_kb(self) -> dict:
        """Get default knowledge base if file not found."""
        return {
            "brand": {
                "name": "Meallion",
                "pronunciation": "Million",
                "founder": "Chef Lambros Vakiaros",
                "description": "Premium Greek food delivery service"
            },
            "faqs": [],
            "greeting": {
                "default": "Γεια σας! Είμαι η Έλενα από το Meallion. Πώς μπορώ να σας βοηθήσω;",
                "english": "Hello! I'm Elena from Meallion. How can I help you today?"
            },
            "closing": {
                "default": "Ευχαριστώ που επικοινωνήσατε με το Meallion. Καλή σας ημέρα!",
                "english": "Thank you for contacting Meallion. Have a great day!"
            }
        }

    def get_tools(self) -> list:
        """Get the list of function tools for this module."""
        return [
            search_knowledge_base,
            get_brand_info,
            get_greeting,
        ]


# Global knowledge base instance
_kb_instance: Optional[KnowledgeBaseTool] = None


def get_kb_instance() -> KnowledgeBaseTool:
    """Get or create knowledge base instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBaseTool()
    return _kb_instance


async def search_knowledge_base(
    query: Annotated[str, "The customer's question or topic to search for"],
    language: Annotated[str, "Language for response: 'en' or 'el'"] = "en",
) -> str:
    """
    Search the Meallion knowledge base for answers to common questions.
    
    NOTE: This is a BACKUP tool. The knowledge base is already injected into your system prompt,
    so you should answer most questions DIRECTLY without calling this tool.
    
    Only use this if you absolutely cannot find the information in your context.
    
    Use this for general inquiries about:
    - Company information (what is Meallion, who founded it)
    - Meal categories (protein boost, signature, plant-based)
    - Delivery and ordering info
    - Product info (heating, storage, packaging)
    - Nutritional info and allergens
    - Pricing and minimum orders
    
    Args:
        query: The question or topic to search
        language: Language code ('en' or 'el') for bilingual responses
        
    Returns:
        The most relevant answer from the knowledge base
    """
    kb = get_kb_instance()
    
    query_lower = query.lower()
    lang = "el" if language.lower() in ("el", "greek", "ελληνικά") else "en"
    
    # First, try to search database items (most up-to-date)
    try:
        if not kb.db_items:
            await kb.load_db_items()
        db_result = kb.search_db_items(query, lang)
        if db_result:
            return db_result
    except Exception as e:
        logger.warning(f"DB search failed, falling back to file: {e}")
    
    if not kb.kb_data:
        return "I couldn't access the knowledge base. How else can I help you?"
    
    # Helper function to get bilingual field
    def get_field(data, fallback=""):
        if isinstance(data, dict):
            return data.get(lang, data.get("en", fallback))
        return data or fallback
    
    # WHAT IS MEALLION / ABOUT
    if any(word in query_lower for word in ["what is meallion", "about meallion", "meallion", "company", "brand", "τι είναι", "εταιρία"]):
        about = kb.kb_data.get("about", {})
        brand = kb.kb_data.get("brand", {})
        what_is = about.get("what_is_meallion", {})
        result = get_field(what_is, get_field(brand.get("one_liner", {})))
        if result:
            logger.info(f"KB match for '{query}': about/what_is_meallion ({lang})")
            return result
    
    # FOUNDERS / WHO MADE IT / OWNER
    if any(word in query_lower for word in ["founder", "chef", "who made", "who started", "lampros", "vakiaros", "owner", "owns", "created", "started by"]):
        about = kb.kb_data.get("about", {})
        founders = about.get("founders", {})
        chef = founders.get("chef", {})
        partner = founders.get("partner", {})
        result = f"Meallion was founded by {chef.get('name', 'Lampros Vakiaros')} ({chef.get('role', 'MasterChef winner')}) and {partner.get('name', 'Theodoros Papaloukas')} ({partner.get('role', 'former basketball player')})."
        logger.info(f"KB match for '{query}': founders")
        return result
    
    # MEAL CATEGORIES / TYPES
    if any(word in query_lower for word in ["protein", "boost", "high protein", "πρωτεΐνη"]):
        categories = kb.kb_data.get("meal_categories", {})
        pb = categories.get("protein_boost", {})
        desc = get_field(pb.get("description", {}), "High protein meals")
        logger.info(f"KB match for '{query}': protein_boost ({lang})")
        return f"{pb.get('name', 'Protein Boost')}: {desc}"
    
    if any(word in query_lower for word in ["signature", "gourmet", "special"]):
        categories = kb.kb_data.get("meal_categories", {})
        sig = categories.get("signature", {})
        logger.info(f"KB match for '{query}': signature")
        return f"{sig.get('name', 'Signature')}: {sig.get('description', 'Gourmet dishes')}"
    
    if any(word in query_lower for word in ["vegetarian", "vegan", "plant", "veggie"]):
        categories = kb.kb_data.get("meal_categories", {})
        pb = categories.get("plant_based_veggie", {})
        logger.info(f"KB match for '{query}': plant_based")
        return f"{pb.get('name', 'Plant-Based')}: {pb.get('description', 'Vegetarian options')}"
    
    if any(word in query_lower for word in ["full taste", "less guilt", "comfort"]):
        categories = kb.kb_data.get("meal_categories", {})
        ft = categories.get("full_taste_less_guilt", {})
        logger.info(f"KB match for '{query}': full_taste")
        return f"{ft.get('name', 'Full Taste')}: {ft.get('description', 'Balanced comfort food')}"
    
    if any(word in query_lower for word in ["categories", "types", "meal types", "what meals", "menu"]):
        categories = kb.kb_data.get("meal_categories", {})
        result = "We have 4 meal categories: "
        result += f"1) Protein Boost - {categories.get('protein_boost', {}).get('description', 'high protein')}, "
        result += f"2) Full Taste, Less Guilt - {categories.get('full_taste_less_guilt', {}).get('description', 'balanced')}, "
        result += f"3) Signature - {categories.get('signature', {}).get('description', 'gourmet')}, "
        result += f"4) Plant-Based - {categories.get('plant_based_veggie', {}).get('description', 'vegetarian')}."
        logger.info(f"KB match for '{query}': meal_categories")
        return result
    
    # HEATING / STORAGE
    if any(word in query_lower for word in ["heat", "microwave", "oven", "warm"]):
        product = kb.kb_data.get("product_info", {})
        heating = product.get("heating_instructions", {})
        logger.info(f"KB match for '{query}': heating")
        return f"Microwave: {heating.get('microwave', '3 min at 600W')}. Oven: {heating.get('oven', '120°C for 20 min')}."
    
    if any(word in query_lower for word in ["storage", "fridge", "keep", "fresh", "how long"]):
        product = kb.kb_data.get("product_info", {})
        storage = product.get("storage", {})
        logger.info(f"KB match for '{query}': storage")
        return f"Fridge: {storage.get('fridge', 'Up to 7 days at 0-4°C')}. {storage.get('freezing', 'Freezing not recommended')}."
    
    # DELIVERY / ORDERING
    if any(word in query_lower for word in ["delivery", "shipping", "deliver", "ship"]):
        ordering = kb.kb_data.get("ordering", {})
        fee = ordering.get("delivery_fee", {})
        logger.info(f"KB match for '{query}': delivery")
        return f"Delivery: {fee.get('5_meals_or_more', 'Free for 5+ meals')}, {fee.get('under_5_meals', '€6 for under 5 meals')}. {ordering.get('delivery_areas', 'Athens area')}."
    
    if any(word in query_lower for word in ["minimum", "order", "how many"]):
        ordering = kb.kb_data.get("ordering", {})
        logger.info(f"KB match for '{query}': minimum_order")
        return f"Minimum order: {ordering.get('minimum_order', '2 meals')}. {ordering.get('recommendation', 'We recommend 5+ for weekly planning')}."
    
    # PRICE
    if any(word in query_lower for word in ["price", "cost", "expensive", "cheap", "worth"]):
        scripts = kb.kb_data.get("call_scripts", {})
        logger.info(f"KB match for '{query}': price")
        return scripts.get("price_objection", "Our meals are competitively priced compared to daily delivery, with better quality and consistency.")
    
    # DIET
    if any(word in query_lower for word in ["diet", "weight loss", "lose weight", "healthy"]):
        scripts = kb.kb_data.get("call_scripts", {})
        logger.info(f"KB match for '{query}': diet")
        return scripts.get("healthy_or_diet", "It's not a diet. It's real food, properly cooked, with balanced portions.")
    
    # ALLERGENS
    if any(word in query_lower for word in ["allergy", "allergen", "gluten", "dairy", "nuts"]):
        allergens = kb.kb_data.get("allergens", {})
        logger.info(f"KB match for '{query}': allergens")
        return f"We list allergens clearly: {', '.join(allergens.get('listed_allergens', ['gluten', 'dairy', 'nuts'])[:5])}. {allergens.get('disclaimer', 'Check packaging for details.')}"
    
    # CONTACT
    if any(word in query_lower for word in ["contact", "phone", "email", "reach"]):
        contact = kb.kb_data.get("contact", {})
        logger.info(f"KB match for '{query}': contact")
        return f"Contact us: Phone {contact.get('phone', '+30 211 9555 451')}, Email {contact.get('email', 'hello@meallion.gr')}, Instagram {contact.get('instagram', '@mealliongr')}."
    
    # MEAL MENU SEARCH (NEW)
    if any(word in query_lower for word in ["menu", "meals", "dishes", "chicken", "beef", "fish", "vegetarian", "γεύματα", "μενού", "κοτόπουλο"]):
        meal_menu = kb.kb_data.get("meal_menu", {})
        if meal_menu:
            logger.info(f"KB match for '{query}': meal_menu")
            return f"We have various meals including chicken, beef/pork, seafood, and vegetarian options. Please check our website or ask me about a specific type."
    
    # No match found - but return helpful info
    logger.info(f"No specific KB match for: {query}")
    brand = kb.kb_data.get("brand", {})
    one_liner = get_field(brand.get("customer_one_liner", {}), "Meallion is high-quality ready-to-eat food designed for eating well consistently.")
    return one_liner


async def get_brand_info(
    aspect: Annotated[str, "What aspect of the brand to get info about: 'name', 'founder', 'description', 'pronunciation'"] = "description",
) -> str:
    """
    Get information about the Meallion brand.
    
    Use this when customers ask about the company, the chef, or general brand information.
    
    Args:
        aspect: Which aspect of brand info to retrieve
        
    Returns:
        The requested brand information
    """
    kb = get_kb_instance()
    
    if not kb.kb_data:
        return "Meallion είναι μια premium υπηρεσία παράδοσης ελληνικού φαγητού."
    
    brand = kb.kb_data.get("brand", {})
    
    aspect_lower = aspect.lower()
    
    if aspect_lower == "name":
        name = brand.get("name", "Meallion")
        pronunciation = brand.get("pronunciation", "Million")
        return f"Το όνομά μας είναι {name}, προφέρεται σαν '{pronunciation}'."
    
    elif aspect_lower == "founder":
        founder = brand.get("founder", "Chef Lambros Vakiaros")
        return f"Το Meallion ιδρύθηκε από τον {founder}."
    
    elif aspect_lower == "pronunciation":
        pronunciation = brand.get("pronunciation", "Million")
        return f"Το Meallion προφέρεται '{pronunciation}'."
    
    else:  # description or default
        description = brand.get("description", "Premium Greek food delivery service")
        founder = brand.get("founder", "Chef Lambros Vakiaros")
        return f"Το Meallion είναι {description}, από τον {founder}."


async def get_greeting(
    language: Annotated[str, "The language for the greeting: 'greek' or 'english'"] = "greek",
) -> str:
    """
    Get the standard greeting phrase.
    
    Use this at the start of a conversation.
    
    Args:
        language: Which language to use
        
    Returns:
        The greeting phrase
    """
    kb = get_kb_instance()
    
    if not kb.kb_data:
        return "Γεια σας! Είμαι η Έλενα από το Meallion. Πώς μπορώ να σας βοηθήσω;"
    
    greetings = kb.kb_data.get("greeting", {})
    
    if language.lower() in ["english", "en"]:
        return greetings.get("english", "Hello! I'm Elena from Meallion. How can I help you today?")
    else:
        return greetings.get("default", "Γεια σας! Είμαι η Έλενα από το Meallion. Πώς μπορώ να σας βοηθήσω;")


async def get_closing(
    language: Annotated[str, "The language for the closing: 'greek' or 'english'"] = "greek",
) -> str:
    """
    Get the standard closing phrase.
    
    Use this at the end of a conversation when the customer is done.
    
    Args:
        language: Which language to use
        
    Returns:
        The closing phrase
    """
    kb = get_kb_instance()
    
    if not kb.kb_data:
        return "Ευχαριστώ που επικοινωνήσατε με το Meallion. Καλή σας ημέρα!"
    
    closings = kb.kb_data.get("closing", {})
    
    if language.lower() in ["english", "en"]:
        return closings.get("english", "Thank you for contacting Meallion. Have a great day!")
    else:
        return closings.get("default", "Ευχαριστώ που επικοινωνήσατε με το Meallion. Καλή σας ημέρα!")
