import os
import requests
from dotenv import load_dotenv
import pprint as pp
from langchain_core.tools import BaseTool, StructuredTool, tool, ToolException
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from datetime import datetime


SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_info(url):
    url = url
    headers = {
        "Authorization": f"{SECTORS_API_KEY}",
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        raise ToolException("We dont have the information about it")


###


def get_current_date_ai():
    """
    Get current date
    """
    return datetime.now().strftime("%Y-%m-%d")


get_current_date_tool = StructuredTool.from_function(
    func=get_current_date_ai,
    name="get_current_date_ai",
    description=f"""
    Get current date, consist of year, month, and day with format YYYY-MM-DD
    """,
)

###


class GetInfoParam(BaseModel):
    ticker: str = Field(description="Company ticker", max_length=4)
    section: str = Field(description="Report section", max_length=20)


def get_info_ai(ticker: str, section: str):
    url = f"https://api.sectors.app/v1/company/report/{ticker.lower()}/?sections={section.lower()}"
    return get_info(url)


get_info_tool = StructuredTool.from_function(
    func=get_info_ai,
    name="get_info_ai",
    description=f"""
    Get certain report section.
    :param ticker: Company ticker from certain company/ticker.
    :param section: Report section such as:
    - Overview: General knowledge about the company, provides information about the company. 
    - Valuation: The extent of the companys wealth/assets, measured with specific criteria.
    - Peers: Companies that have a similar market share, including their respective market capitalizations.
    - Future: Analysis of the company's current financial state and potential future prospects.
    - Technical: A form of security analysis that utilizes price and volume data, usually displayed graphically in charts.
    - Financial: The company's revenue and earnings.
    - Dividend: Stock payout ratings.
    - Management: The managers/chiefs present at BRI.
    - Ownership: Distribution of the companys ownership.
    if user want to get overall information about company, use all of the listed section.
    KEYWORD: company information, company report, company overview
    """,
    args_schema=GetInfoParam,
)

###


class BestNCompanyOfAnyYearParam(BaseModel):
    classifications: str = Field(
        description="Classifications such as Earning, Market Cap, Revenue, etc."
    )
    sub_sector: str = Field(
        description="Company sub_sector type like Banks, Basic Material, Financing Services, etc."
    )
    n: int = Field(description="Top n company")
    year: int = Field(
        description="Year. Get current year by using get_current_date_ai() and get the Year"
    )


def best_n_company_of_any_year_ai(
    classifications: str, sub_sector: str, n: int, year: int
):
    url = f"https://api.sectors.app/v1/companies/top/?classifications={'_'.join(classifications.lower().split())}&n_stock={n}&year={year}&sub_sector={'-'.join(sub_sector.lower().split())}"
    return get_info(url)


best_n_company_of_any_year_tool = StructuredTool.from_function(
    func=best_n_company_of_any_year_ai,
    name="best_n_company_of_any_year_ai",
    description=f"""
    Get list of IDX companies in a given year that ranks top on a specified dimension 
    :classifications: such as Earning, Market Cap, Revenue, etc.
    :sub_sector: Company sub_sector type like Banks, Basic Material, Financing Services, etc.
    :n: Top n company
    :year: Year. Get current year by using get_current_date_ai() function
    KEYWORD: top company, ranking, best company
    """,
    args_schema=BestNCompanyOfAnyYearParam,
)

###


class BestCompanyTradedStockByTransactionVolumeParam(BaseModel):
    start_date: str = Field(
        description="Start date of the period of time. format: yyyy-mm-dd."
    )
    end_date: str = Field(
        description="End date of the period of time. format: yyyy-mm-dd. date must be greater than start date."
    )
    sub_sector: str = Field(
        description="Sub sector of the company. all availabe sub sector. if not provided, make as '' (empty string)",
        default="",
    )
    n: int = Field(description="Number of top company.", default=5)


def best_company_traded_stock_by_transaction_volume_ai(
    start_date: str,
    end_date: str,
    sub_sector: str = "",
    n: int = 5,
):
    """
    Get n companies of most traded stocks based on transaction volume in certain period of time.
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    print(start_date, end_date)
    if sub_sector == "":
        url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={n}"
    else:
        url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={n}&sub_sector={'-'.join(sub_sector.lower().split())}"
    print(url)
    return get_info(url)


best_company_traded_stock_by_transaction_volume_tool = StructuredTool.from_function(
    func=best_company_traded_stock_by_transaction_volume_ai,
    name="best_company_traded_stock_by_transaction_volume_ai",
    description=f"""
    Get n companies of most traded stocks based on transaction volume in certain period of time.
    :param start_date: Start date of the period of time. get the value by using get_current_date_tool. format: yyyy-mm-dd.
    :param end_date: End date of the period of time. get the value by using get_current_date_tool. format: yyyy-mm-dd. date must be greater than start date.
    :param sub_sector: Sub sector of the company.
    :param n: Number of top company.
    KEYWORD: most traded stocks, transaction volume
    """,
    args_schema=BestCompanyTradedStockByTransactionVolumeParam,
)

###


class DailyTransactionParam(BaseModel):
    ticker: str = Field(description="Company ticker", max_length=4)
    start_date: str = Field(
        description="Start date, format YYYY-mm-dd, use get_current_date_ai()",
        max_length=10,
    )
    end_date: str = Field(
        description="End date, format YYYY-mm-dd, use get_current_date_ai()",
        max_length=10,
    )


def get_daily_transaction_ai(ticker: str, start_date: str, end_date: str):
    """
    Return daily transaction data of a given ticker on a certain interval.
    """
    url = (
        f"https://api.sectors.app/v1/daily/{ticker}/?start={start_date}&end={end_date}"
    )
    return get_info(url)


get_daily_transaction_tool = StructuredTool.from_function(
    func=get_daily_transaction_ai,
    name="get_daily_transaction_ai",
    description=f"""
    Return daily transaction data of a given ticker on a certain interval.
    :param ticker: Company ticker from certain company/ticker.
    :param start_date: Start date, format YYYY-mm-dd, use get_current_date_ai() function
    :param end_date: End date, format YYYY-mm-dd, use get_current_date_ai() function
    KEYWORD: daily transaction, closing prices, volume, market cap
    """,
    args_schema=DailyTransactionParam,
)

###

tools = [
    get_current_date_tool,
    best_n_company_of_any_year_tool,
    best_company_traded_stock_by_transaction_volume_tool,
    get_daily_transaction_tool,
    get_info_tool,
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            Respond logically and according to the tools provided. Answer it as humanly as possible.
            if user ask about general information about a company, call tools with all possible sections and coclude it with number as proof.
            if ask about date, use get_current_date_ai tool and figure it out about yesterday, tomorrow, or n of days ago with format yyyy-mm-dd.
            use get_info_ai() to get information about a company.
            
            if tools doenst have the parameter, dont force it to the tools.
            IMPORTANT: ADJUST REPLY LANGUAGE TO THE HUMAN INPUT, EX: INDONESIA, JAPANESE, GERMAN
            """,
        ),
        ("human", "{input}"),
        # msg containing previous agent tool invocations
        # and corresponding tool outputs
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm = ChatGroq(
    temperature=0,
    model_name="llama3-groq-70b-8192-tool-use-preview",
    groq_api_key=GROQ_API_KEY,
)


agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# query_0 = "Give me the company leader of BBRI"
# query_0_1 = "give me to 5 Banks companies in 3 years ago"
query_1 = "What are the top 5 company by transaction volume on the first of this month?"
query_2 = "What are the most traded stock yesterday?"
query_3 = (
    "What are the top 7 most traded stocks between 6th June to 10th June this year?"
)
query_4 = "What are the top 3 companies by transaction volume over the last 7 days?"  # gak konsisten atara pakai angka dan tidak
query_5 = "Based on the closing prices of BBCA between 1st and 30th of June 2024, are we seeing an uptrend or downtrend? Try to explain why."
query_6 = "What is the company with the largest market cap between BBCA and BREN? For said company, retrieve the email, phone number, listing date and website for further research."
query_7 = "What is the performance of GOTO (symbol: GOTO) since its IPO listing?"
query_8 = "If i had invested into GOTO vs BREN on their respective IPO listing date, which one would have given me a better return over a 90 day horizon?"


queries = [query_2]

for query in queries:
    print("Question:", query)
    result = agent_executor.invoke({"input": query})
    print("Answer:", "\n", result["output"], "\n\n======\n\n")
