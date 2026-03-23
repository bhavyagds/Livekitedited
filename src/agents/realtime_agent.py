import logging
import os
import asyncio
from typing import Annotated
from dotenv import load_dotenv

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    multimodal,
    llm,
)
from livekit.plugins import openai
from src.agents.prompts import get_system_prompt_async, get_greeting, get_agent_language
from src.agents.tools import OrderLookupTool, SupportTicketTool, KnowledgeBaseTool

load_dotenv()

logger = logging.getLogger("realtime-agent")
logger.setLevel(logging.INFO)

class ElenaFunctionContext(llm.FunctionContext):
    """Function context for the Elena voice agent."""
    def __init__(self):
        super().__init__()
        self.order_tool = OrderLookupTool()
        self.ticket_tool = SupportTicketTool()
        self.kb_tool = KnowledgeBaseTool()

    @llm.ai_callable(description="Look up order details for a customer.")
    async def lookup_order(self, order_number: Annotated[str, "The order number to look up"]):
        logger.info(f"Looking up order: {order_number}")
        return await self.order_tool.lookup_order(order_number)

    @llm.ai_callable(description="Create a support ticket for a customer issue.")
    async def create_support_ticket(
        self, 
        name: Annotated[str, "Customer's full name"], 
        phone: Annotated[str, "Customer's phone number"], 
        email: Annotated[str, "Customer's email address"], 
        issue: Annotated[str, "Description of the issue"]
    ):
        logger.info(f"Creating support ticket for {name}")
        return await self.ticket_tool.create_support_ticket(name, phone, email, issue)

    @llm.ai_callable(description="Search the Meallion knowledge base for information.")
    async def search_knowledge_base(self, query: Annotated[str, "Topic or question to search for"]):
        logger.info(f"Searching KB for: {query}")
        return await self.kb_tool.search_knowledge_base(query)

async def entrypoint(ctx: JobContext):
    try:
        logger.info(f"Connecting to room {ctx.room.name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # Wait for the first participant to join
        participant = await ctx.wait_for_participant()
        logger.info(f"Starting Elena Realtime agent for participant {participant.identity}")

        # Load Elena's persona and greeting dynamically
        language = get_agent_language()
        system_prompt = await get_system_prompt_async(language)
        system_prompt = system_prompt[:1000] # TRUNCATE FOR WEBSOCKET LIMIT
        greeting = get_greeting(language)

        logger.info(f"Loaded persona for {language} ({len(system_prompt)} chars)")

        # Initialize the OpenAI Realtime model explicitly
        model = openai.realtime.RealtimeModel(
            model="gpt-4o-realtime-preview-2024-10-01",
            instructions=system_prompt,
            modalities=["audio", "text"],
            voice="alloy", 
        )

        agent = multimodal.MultimodalAgent(
            model=model,
            fnc_ctx=ElenaFunctionContext(),
        )
        agent.start(ctx.room, participant)

        # Send an initial message to prompt the agent to start speaking
        session = agent.model.sessions[0]
        session.conversation.item.create(
            llm.ChatMessage(
                role="user",
                content=f"Please say your greeting now: '{greeting}'"
            )
        )
        session.response.create()

        logger.info("Elena Realtime agent fully started and prompted to greet")
    except Exception as e:
        logger.error(f"FATAL ERROR in entrypoint: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="",
            api_key=os.environ.get("LIVEKIT_API_KEY"),
            api_secret=os.environ.get("LIVEKIT_API_SECRET"),
            ws_url=os.environ.get("LIVEKIT_URL"),
        )
    )
