import openai
import os
import phi
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import phi.api
from phi.agent import Agent
from dotenv import load_dotenv
from phi.playground import Playground, serve_playground_app

# Load the OpenAI API key from the .env file
load_dotenv()
phi.api = os.getenv("PHI_API_KEY")    

# Create an agent that can search the web
web_search_agent = Agent(
    name="Web Search Agent",
    role= "Search the web for the information", 
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Alway include sources"],
    show_tools_calls=True,
    markdown=True,
)

# Create an agent that can get financial information
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

app= Playground(agents=[web_search_agent, financial_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True)

