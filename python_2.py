import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import (
    create_tool_calling_agent,
    AgentExecutor,
)
from datetime import datetime

import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler


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
        return {"error": "Failed to get data", "reason": e}


###


@tool
def get_average(myList):
    """
    Get average of a list of numbers provided
    """
    return sum(myList) / len(myList)


###


@tool
def get_current_date_ai():
    """
    Get current date
    """
    return datetime.now().strftime("%Y-%m-%d")


@tool
def get_available_subsectors_ai():
    """
    Get available subsectors of companies in sectors app
    """
    url = "https://api.sectors.app/v1/subsectors/"
    return get_info(url)


@tool
def get_info_ai(ticker: str, section: str):
    """
    :param section: Get certain report section such as overview, financials, feature, peers etc
    - Overview: General knowledge about the company, provides information about the company. [industry, subsector, market cap, sector, ticker, addressm employee num, website, phone]
    - Valuation: The extent of the companys wealth/assets, measured with specific criteria. [pe, pb, ps, peg ratio]
    - Peers: Companies that have a similar market share, including their respective market capitalizations. [group name, peers companies]
    - Future: Analysis of the company's current financial state and potential future prospects. [forecast, growth]
    - Financials: The company's revenue and earnings. [tax, revenue, earnings, debt, asset, profits, equity, liabilities]
    - Dividend: Stock payout ratings. [total yield, total divident]
    - Management: The managers/chiefs present at BRI. [executive, shareholder]
    - Ownership: Distribution of the companys ownership. [major shareholder, ownership percentage, monthly net transaction]
    :param ticker: company/ticker such as BBRI, BMRI, BBCA, etc

    if user asked about information inside the section, get all of the section available
    """
    url = f"https://api.sectors.app/v1/company/report/{ticker.lower()}/?sections={section.lower()}"
    return get_info(url)


@tool
def best_n_company_of_any_year_ai(
    classifications: str, sub_sector: str, n: int, year: int
):
    """
    :param sub_sector: Get top n company in certain year by classifiying company sub_sector type like Banks, Basic Material, Financing Services, etc.
    :param classification: classified it by classifications [dividend yield, earnings, market cap, revenue, total dividend]
    """
    url = f"https://api.sectors.app/v1/companies/top/?classifications={'_'.join(classifications.lower().split())}&n_stock={n}&year={year}&sub_sector={'-'.join(sub_sector.lower().split())}"
    return get_info(url)


@tool
def best_company_traded_stock_by_transaction_volume_ai(
    start_date: str,
    end_date: str,
    sub_sector: str = "",
    n: int = 5,
):
    """
    Get top n of most traded company by transaction volume in certain period of time.
    :param start_date: Start date of the period of time. format: yyyy-mm-dd.
    :param end_date: End date of the period of time. format: yyyy-mm-dd. date must be greater than start date.
    :param sub_sector: Sub sector of the company. all availabe sub sector can be get by calling get_available_subsectors_ai
    :param n: Number of top company. default is 5
    :return: List of top n company by transaction volume (with company name, ticker, and transaction volume) sorted by transaction volume ascending.
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


@tool
def get_daily_transaction_ai(ticker: str, start_date: str, end_date: str):
    """
    Return daily transaction data of a given ticker on a certain interval
    contains closing prices, volume, and market cap
    :param ticker: company ticker
    :param start_date: start date of the interval
    :param end_date: end date of the interval
    """
    url = (
        f"https://api.sectors.app/v1/daily/{ticker}/?start={start_date}&end={end_date}"
    )
    return get_info(url)


@tool
def get_performance_of_company_since_ipo_listing_ai(ticker: str):
    """
    Get the performance of the company since its IPO listing.
    """
    url = f"https://api.sectors.app/v1/listing-performance/{ticker}/"
    return get_info(url)


@tool
def get_revenue_and_cost_segment(ticker: str):
    """
    Get revenue and cost segment of a company [revenue breakdown, value, source, target]
    :param ticker: company ticker
    """
    url = f"https://api.sectors.app/v1/company/financials/{ticker}/"
    return get_info(url)


tools = [
    get_current_date_ai,
    get_available_subsectors_ai,
    get_info_ai,
    best_n_company_of_any_year_ai,
    best_company_traded_stock_by_transaction_volume_ai,
    get_performance_of_company_since_ipo_listing_ai,
    get_daily_transaction_ai,
    get_revenue_and_cost_segment,
    get_average,
]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            Respond logically and according to the tools provided. Answer it as humanly as possible.
            if user doesnt provide subsector, use the default value.
            if ask about date, use get_current_date_ai tool and figure it out about yesterday, tomorrow, or n of days ago with format yyyy-mm-dd
            if ask about average, make a list of asked number, and use get_average tool to return final value.
            
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

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st.write("🧠 thinking...")
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt}, callback_handler=st_callback
        )
        st.write(response)
