from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


web_search_agent = Agent(
    name="Web Search Agent",
    role= "Search the web for the information", 
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Alway include sources"],
    show_tools_calls=True,
    markdown=True,
)

financial_agent= Agent(
    name="Financial Agent",
    role="Get financial information",
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, stock_fundamentals=True,
                analyst_recommendations=True, company_news=True)],
    instructions=["Use Tables to display the data"],
    show_tools_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[web_search_agent, financial_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instrtuctions=["Always include sources", "Use Tables to display the data"],
    markdown=True,
    show_tools_calls=True,
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news on NVDA")


