from dotenv import load_dotenv
from phoenix.otel import register

load_dotenv()

tracer_provider = register(
    project_name="openai-agents",
    auto_instrument=True,
)

from pprint import pprint
from textwrap import dedent

from agents import Agent, Runner, TResponseInputItem, WebSearchTool
from agents.model_settings import ModelSettings

qa_agent = Agent(
    name="Investing Q&A Agent",
    instructions=dedent("""You are Warren Buffett. You are answering questions about investing."""),
    model="gpt-4.1",
)

research_agent = Agent(
    name="Financial Search Agent",
    instructions=dedent(
        """You are a research assistant specializing in financial topics. Given a stock ticker, use web search to retrieve up‑to‑date context and produce a short summary of at most 50 words. Focus on key numbers, events, or quotes that will be useful to a financial analyst."""
    ),
    model="gpt-4.1",
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required", parallel_tool_calls=True),
)

orchestrator_agent = Agent(
    name="Routing Agent",
    instructions=dedent(
        """You are a senior financial analyst. Your task is to handoff to the appropriate agent or tool."""
    ),
    model="gpt-4.1",
    handoffs=[
        research_agent,
        qa_agent,
    ],
)

input_items: list[TResponseInputItem] = []

while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    input_item = {"content": user_input, "role": "user"}
    input_items.append(input_item)
    orchestrator = Runner.run_sync(orchestrator_agent, input_items)
    orchestrator_output = orchestrator.final_output
    pprint(orchestrator.last_agent)
    pprint(orchestrator_output)
    input_items = orchestrator.to_input_list()
