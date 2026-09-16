from dotenv import load_dotenv

load_dotenv()
from agents import Agent, Runner, WebSearchTool

agent = Agent(
    name="Finance Agent",
    instructions="You are a finance agent that can answer questions about stocks. Use web search to retrieve up‑to‑date context. Then, return a brief, concise answer that is one sentence long.",
    tools=[WebSearchTool()],
    model="gpt-4.1-mini",
)
from phoenix.otel import register

register(project_name="finance-agent", auto_instrument=True)
result = Runner.run_sync(
    agent,
    "What are the latest major developments involving Apple?",
)

print(result.final_output)