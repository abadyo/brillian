import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime
import datetime as dt
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
import pandas as pd
import json
import matplotlib.pyplot as plt
import altair as alt
import pprint as pp
import ast
from thefuzz import fuzz
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.tools import DuckDuckGoSearchRun

# SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# get current directory
current_dir = os.path.dirname(__file__)
subsector = os.path.join(current_dir, "subsector.txt")

if "sectors_api_key" not in st.session_state:
    st.session_state["sectors_api_key"] = ""
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""


# query_0 = "BBRI Overview"
query_1 = "What are the top 5 company by transaction volume on the first of this month?"
query_2 = "What are the most traded stock yesterday?"
query_3 = (
    "What are the top 7 most traded stocks between 6th June to 10th June this year?"
)
query_4 = "What are the top 3 companies by transaction volume over the last 7 days?"  # gak konsisten atara pakai angka dan tidak
query_5 = "Based on the closing prices of BBCA between 1st and 30th of June 2024, are we seeing an uptrend or downtrend? Try to explain why."
query_6 = "What is the company with the largest market cap between BBCA and BREN? For said company, retrieve the email, phone number, listing date and website for further research."
query_7 = "What is the performance of GOTO  since its IPO listing?"
query_8 = "If i had invested into GOTO vs BREN on their respective IPO listing date, which one would have given me a better return over a 90 day horizon?"
query_9 = "Give me the 2nd place contact information of the top 3 companies by transaction volume on the 1st of july 2024"
query_10 = "How much is the average of BBCA close price from 1st of august 2024 until 3 august 2024?"
query_11 = "Show me the graph of BBRI closing price from 1st of august 2024 until 7 august 2024."
query_13 = "How many companies are in the subsector of oil and gas?"
query_14 = "What is the ABCD.JK overview?"
query_15 = "Whos the top 5 total dividend in basic materials subsector"

queries = [
    query_1,
    query_2,
    query_3,
    query_4,
    query_5,
    query_6,
    query_7,
    query_8,
    query_9,
    query_10,
    query_11,
    query_13,
    query_14,
    query_15,
]

with st.sidebar:
    sectors_api_key = st.text_input(
        "Sectors API Key", key="sectors_api_key", type="password"
    )
    groq_api_key = st.text_input("Groq API Key", key="groq_api_key", type="password")

    button = st.button("Reset API Keys")

    if button:
        del st.session_state["sectors_api_key"]
        del st.session_state["groq_api_key"]

    "[Get Sectors API Key](https://sectors.app/api)"
    "[Get Groq API Key](https://console.groq.com/keys)"

    for i in queries:
        st.write(i)


def get_info(url):
    st.write("Fetching data from sectors..")
    print(url)
    headers = {
        "Authorization": f"{st.session_state['sectors_api_key']}",
    }
    try:
        response = requests.get(url, headers=headers)
        print(response)
        if response.status_code != 200:
            return None
        # print(response.json())
        return response.json()
    except requests.exceptions.HTTPError as e:
        return None


def fuzzy_search_company_name(x):
    """
    transform company name into symbol with fuzzy search
    :param x: company name
    """
    with open(os.path.join(current_dir, "company.json")) as f:
        dict = {}
        data = json.load(f)
        for i in data:
            dict[i["symbol"]] = i["company_name"]

        max = 0
        for key, value in dict.items():
            similarity = fuzz.ratio(x, value)
            if similarity >= 90:
                max = similarity
                result = key
        if max < 90:
            return None
        return result


def fuzzy_search_symbol(x):
    """
    transform company name into symbol with fuzzy search
    :param x: company name
    """

    if x[-3:] == ".JK":
        pass
    else:
        x = x + ".JK"

    with open(os.path.join(current_dir, "company.json")) as f:
        dict = {}
        data = json.load(f)
        for i in data:
            dict[i["symbol"]] = i["company_name"]

        max = 0
        for key, value in dict.items():
            similarity = fuzz.ratio(x, key)
            if similarity >= 90:
                max = similarity
                result = key
        if max < 90:
            return None
        return result


def fuzzy_search_subsector(x):
    with open(os.path.join(current_dir, "subsector.txt"), "r") as file:
        sectors = file.read().splitlines()
        max = 0
        # fuzzy matching for the highest match
        for sector in sectors:
            score = fuzz.ratio(x, sector)
            if score > max:
                max = score
                match = sector
        return match


def is_it_holiday(x):
    """
    Check if the date is holiday or not
    :param x: date
    """
    x = datetime.strptime(x, "%Y-%m-%d")
    a = x.weekday()

    if a == 5:
        x -= dt.timedelta(days=1)
    elif a == 6:
        x += dt.timedelta(days=1)

    x = x.strftime("%Y-%m-%d")
    # print(x)
    return x


##################################################################


@tool
def get_company_information_ai(
    section: str, symbol: str = None, company_name: str = None
):
    """
    :param symbol: company symbol that ends with .JK
    :param section: Get company information of any section from:
        - Overview: General knowledge about the company, provides information about the company. [industry, subsector, market cap, sector, symbol, addressm employee num, website, phone]
        - Valuation: The extent of the companys wealth/assets, measured with specific criteria. [pe, pb, ps, peg ratio]
        - Peers: Companies that have a similar market share, including their respective market capitalizations. [group name, peers companies]
        - Future: Analysis of the company's current financial state and potential future prospects. [forecast, growth]
        - Financials: The company's revenue and earnings. [tax, revenue, earnings, debt, asset, profits, equity, liabilities]
        - Dividend: Stock payout ratings. [total yield, total divident]
        - Management: The managers/chiefs present at BRI. [executive, shareholder]
        - Ownership: Distribution of the companys ownership. [major shareholder, ownership percentage, monthly net transaction]
    :param company_name: company name that ends with Tbk.
    """
    st.write("Searching for company information..")
    # print("hello1")
    if symbol:
        symbol = fuzzy_search_symbol(symbol)
        # print(symbol)
        if symbol is None:
            return {"error": "Symbol not found."}
    elif company_name is not None and company_name != "":
        symbol = fuzzy_search_company_name(company_name)
        # print(symbol)
        if symbol is None:
            return {"error": "Company name not found."}
    # print("hello2")
    url = f"https://api.sectors.app/v1/company/report/{symbol}/?sections={section.lower()}"

    x = get_info(url)
    # print(x)
    return x


@tool
def get_top_transaction_volume_ai(
    start_date: str,
    end_date: str,
    sub_sector: str = None,
    n: int = 5,
):
    """
    Get top n of most traded company by transaction volume in certain period of time.
    :param start_date: Start date of the period of time. format: yyyy-mm-dd.
    :param end_date: End date of the period of time. format: yyyy-mm-dd. date can be same as start date.
    :param sub_sector: Sub sector of the company. all availabe sub sector can be get by calling get_available_subsectors_ai
    :param n: Number of top company. default is 5

    :return: List of top n company by transaction volume {date: [symbol, company_name, valume, price]}
    """
    st.write("Searching for top transaction volume..")
    messege = ""

    if sub_sector:
        sub_sector = fuzzy_search_subsector(sub_sector)
        if sub_sector is None:
            return {"error": "Subsector not found."}

    def sum_volume_price(x):
        result = {}
        for date in x:
            for company in x[date]:
                symbol = company["symbol"]
                company_name = company["company_name"]
                volume = company["volume"]
                price = company["price"]
                if company_name not in result:
                    result[company_name] = {
                        "volume": 0,
                        "price": [],
                        "date-volume-history": {},
                    }
                result[company_name]["volume"] += volume
                result[company_name]["price"].append(price)
                result[company_name]["symbol"] = symbol
                result[company_name]["date-volume-history"][date] = volume
        result = dict(
            sorted(result.items(), key=lambda item: item[1]["volume"], reverse=True)
        )
        # filter top n
        # result = dict(list(result.items())[:n])
        return result

    a = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    b = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")

    start_date = is_it_holiday(start_date)
    end_date = is_it_holiday(end_date)

    # check if date change
    if a != start_date or b != end_date:
        messege = f"Date changed to {start_date} - {end_date} because the original date is holiday"

    if sub_sector == "" or sub_sector is None:
        url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={n}"
    else:
        url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={n}&sub_sector={'-'.join(sub_sector.lower().split())}"
    # print(url)
    x = get_info(url)
    if x is None:
        return {"error": "Data not found."}
    z = sum_volume_price(x)
    return [z, messege]


@tool
def get_daily_transaction_ai(
    start_date: str,
    end_date: str,
    component: str,
    symbol: str = None,
    show_graph: bool = False,
    company_name: str = None,
):
    """
    Return daily transaction data of a given symbol on a certain interval. contains closing prices, volume, and market cap
    :param symbol: company symbol that ends with .JK. if company name is provided, symbol will be generated using fuzzy search.
    :param start_date: start date of the interval
    :param end_date: end date of the interval, date can be same as start date.
    :param show_graph: show line graph of the data.
    :param company_name: company name that ends with Tbk.
    :param component: component of the company data. can be close, volume, market_cap

    :return: List of daily transaction data [symbol, date, close, volume, market_cap]
    """
    st.write("Searching for daily transaction data..")

    messege = ""

    if symbol:
        symbol = fuzzy_search_symbol(symbol)
        if symbol is None:
            return {"error": "Symbol not found."}
    if company_name is not None and company_name != "":
        symbol = fuzzy_search_company_name(company_name)
        if symbol is None:
            return {"error": "Company name not found."}

    a = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    b = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")

    start_date = is_it_holiday(start_date)
    end_date = is_it_holiday(end_date)

    if a != start_date or b != end_date:
        messege = f"Date changed to {start_date} - {end_date} because the original date is holiday"

    url = (
        f"https://api.sectors.app/v1/daily/{symbol}/?start={start_date}&end={end_date}"
    )
    if show_graph:
        st.write("Generating graph..")

        x = get_info(url)
        df = pd.DataFrame(x)
        # print(df)

        first_date = df.iloc[0]
        last_date = df.iloc[-1]

        market_cap = df["market_cap"].tolist()
        volume = df["volume"].tolist()
        close = df["close"].tolist()

        if component == "close":
            c = (
                alt.Chart(df)
                .mark_line()
                .encode(
                    x=alt.X("date"),
                    y=alt.Y(
                        "close:Q",
                        scale=alt.Scale(
                            domain=[
                                (min((close)) - (max(close) - min(close) // 2)),
                                (max((close)) + (max(close) - min(close) // 2)),
                            ]
                        ),
                    ),
                    color="symbol",
                    tooltip=["date", "close"],
                )
            )
            return st.altair_chart(c, use_container_width=True)
        elif component == "volume":
            c = (
                alt.Chart(df)
                .mark_line()
                .encode(
                    x=alt.X("date"),
                    y=alt.Y(
                        "volume:Q",
                        scale=alt.Scale(
                            domain=[
                                (min((volume)) - (max(volume) - min(volume) // 2)),
                                (max((volume)) + (max(volume) - min(volume) // 2)),
                            ]
                        ),
                    ),
                    color="symbol",
                    tooltip=["date", "volume"],
                )
            )
            return st.altair_chart(c, use_container_width=True)
        elif component == "market_cap":
            c = (
                alt.Chart(df)
                .mark_line()
                .encode(
                    x=alt.X("date"),
                    y=alt.Y(
                        "market_cap",
                        scale=alt.Scale(
                            domain=[
                                (
                                    min((market_cap))
                                    - (max(market_cap) - min(market_cap) // 2)
                                ),
                                (
                                    max((market_cap))
                                    + (max(market_cap) - min(market_cap) // 2)
                                ),
                            ]
                        ),
                        axis=alt.Axis(
                            labelExpr="datum.value / 1000000000000 + 'B'", format="~s"
                        ),
                    ),
                    color="symbol",
                    tooltip=["date", "market_cap"],
                )
            )
            st.write(st.altair_chart(c, use_container_width=True))
            # return st.altair_chart(c, use_container_width=True)
            return [df, messege]
        else:
            return None
    else:
        x = get_info(url)
        if x is None or len(x) == 0:
            return {"error": "Data not found."}
        df = pd.DataFrame(x)
        # get first and last date with its closing price, volume, and market cap
        first_date = df.iloc[0]
        last_date = df.iloc[-1]

        # convert all market cap, volume, and close into list
        market_cap = df["market_cap"].tolist()
        volume = df["volume"].tolist()
        close = df["close"].tolist()

        dicti = {
            "symbol": symbol,
            "first_close": first_date["close"],
            "first_volume": first_date["volume"],
            "first_market_cap": first_date["market_cap"],
            "last_close": last_date["close"],
            "last_volume": last_date["volume"],
            "last_market_cap": last_date["market_cap"],
            "market cap values": market_cap,
            "volume values": volume,
            "close values": close,
        }
        return [
            dicti,
            messege
            + " Use get_average_from_a_list_of_number to calculate the average of close, market_cap and use get_change_percentage_from_two_number to get change percentage. Dont make things up!.",
        ]


@tool
def get_performance_of_company_since_ipo_listing_ai(symbol: str):
    """
    Get the performance of the company since its IPO listing.
    :param symbol: company symbol that ends with .JK

    :return: Performance of the company since its IPO listing [symbol, change_7d, change_30d, change_90d, change_365d]
    """
    st.write("Searching for company performance since IPO listing..")
    url = f"https://api.sectors.app/v1/listing-performance/{symbol}/"
    x = get_info(url)
    if x is None:
        return {"error": "Data not found."}

    # convert to %
    if x["chg_7d"] is not None:
        x["chg_7d"] = f"{x['chg_7d'] * 100}%"
    if x["chg_30d"] is not None:
        x["chg_30d"] = f"{x['chg_30d'] * 100}%"
    if x["chg_90d"] is not None:
        x["chg_90d"] = f"{x['chg_90d'] * 100}%"
    if x["chg_365d"] is not None:
        x["chg_365d"] = f"{x['chg_365d'] * 100}%"

    x["change_in_7_days"] = x.pop("chg_7d")
    x["change_in_30_days"] = x.pop("chg_30d")
    x["change_in_90_days"] = x.pop("chg_90d")
    x["change_in_365_days"] = x.pop("chg_365d")

    # print(x)
    # change float to percentage exept symbol
    return x


@tool
def subsector_aggregated_detail_statistics(sub_sector: str, section: str = None):
    """
    Get aggregated statistics of a subsector
    :param sub_sector: subsector name
    :param section: section of the data:
        - companies: list of companies in the subsector with best company [top market cap, top growth, top profit, top revenue]
        - Growth: Growth of the subsector [growth, yoy earning, yoy revenue, growth forecast]
        - Market_cap: Market cap of the subsector [market cap, average market cap, querter performance]
        - Stability: Stability of the subsector [max_drawdow, rsd_close]
        - Statistics: Statistics of the subsector [total company, median_pe, avg_pe, in_pe, max_pe]
        - Valuation: Valuation of the subsector [pe, pb, ps]

    :return: Aggregated statistics of a subsector
    """
    # print(sub_sector)
    st.write("Searching for subsector.statistics..")
    sub_sector = fuzzy_search_subsector(sub_sector)
    if sub_sector is None:
        return {"error": "Subsector not found."}

    if section:
        url = f"https://api.sectors.app/v1/subsector/report/{sub_sector}/?sections={section.lower()}"
    else:
        url = f"https://api.sectors.app/v1/subsector/report/{sub_sector}/"
    x = get_info(url)
    if x is None:
        return {"error": "Data not found."}
    return x


@tool
def top_subsector_ranked_by_dividend_yield__market_cap__total_dividend__revenue__earnings(
    classification: str, year: str, sub_sector: str, n: int = 5
):
    """
    Return list of company n a given year that ranks top on a specified dimension (total_dividend, revenue, earnings, market_cap, dividend_yield)
    :param classification: classification of the company. can be total_dividend, revenue, earnings, market_cap, dividend_yield
    :param year: year of the data
    :param n: number of top company
    :param sub_sector: subsector name
    """
    st.write("Searching for top subsector ranking..")
    year = str(year)

    if classification not in [
        "total_dividend",
        "revenue",
        "earnings",
        "market_cap",
        "dividend_yield",
    ]:
        return {"error": "Invalid classification."}

    if sub_sector:
        sub_sector = fuzzy_search_subsector(sub_sector)
        if sub_sector is None:
            return {"error": "Subsector not found."}

    url = f"https://api.sectors.app/v1/companies/top/?classifications={classification}&n_stock={n}&year={year}&sub_sector={sub_sector}"
    x = get_info(url)
    if x is None:
        return {"error": "Data not found."}
    return x


@tool
def get_change_percentage_from_two_number(first, last):
    """
    Get change percentage between two numbers
    :param first: first number
    :param last: last number

    extract number from string
    """
    try:
        last = float(last)
        first = float(first)
        return ((last - first) / first) * 100
    except Exception as e:
        return {"error": "Cannot calculate the change percentage."}


@tool
def get_average_from_a_list_of_number(numbers):
    """
    Get average of a list of numbers
    :param numbers: list of numbers, format: [1, 2, 3, 4, 5].
    """
    # convert
    st.write("Averaging numbers...")
    try:
        if type(numbers) == str:
            numbers = ast.literal_eval(numbers)
        numbers = [float(x) for x in numbers]
        return sum(numbers) / len(numbers)
    except Exception as e:
        return {"error": "Cannot calculate the average."}


##################################################################

tools = [
    get_company_information_ai,
    get_top_transaction_volume_ai,
    get_daily_transaction_ai,
    get_performance_of_company_since_ipo_listing_ai,
    get_change_percentage_from_two_number,
    get_average_from_a_list_of_number,
    subsector_aggregated_detail_statistics,
    top_subsector_ranked_by_dividend_yield__market_cap__total_dividend__revenue__earnings,
    DuckDuckGoSearchRun(name="search"),
]


memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=3, human_prefix="human", ai_prefix="ai"
)


@st.cache_resource
def LLM_Chat(key):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                Today is {datetime.now().strftime("%Y-%m-%d")}. Use this as time context.
                Respond logically and according to the tools provided. Answer it as humanly as possible.
                
                this AI only provide information about indonesia index stock market. If user ask about other country other than indonesia, say sorry, this AI only provide information about indonesia index stock market.
                
                if human ask about company, use get_company_information_ai with its symbol either from input or output of other tools to get the information about the company.
                If two different tools were used, use the first tool and then, use the output to second tool get final result, and so on.
                
                answer as detailed as possible with included number if available, but don't call the plot function if it is not necessary.
                
                Dont make random number up, use the tools provided and information from the previous output to calculate the answer.
                use get_change_percentage_from_two_number to calculate the change percentage between two numbers. and use get_average_from_a_list_of_number to calculate the average of a list of numbers. 
                
                for last resort or available tools doesntt provide any answer, use DuckDuckGoSearchRun to search the answer.
                """,
            ),
            ("ai", "{chat_history}"),
            ("human", "{input}"),
            # msg containing previous agent tool invocations
            # and corresponding tool outputs
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    llm = ChatGroq(
        temperature=0, model_name="llama-3.1-70b-versatile", groq_api_key=key
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors="Do not print literal actions. Just do the action needed and return a value from the prevous output",
        memory=memory,
    )
    return agent_executor


##################################################################

# Quey test

# query_0 = "BBRI Overview"
# query_1 = "What are the top 5 company by transaction volume on the first of this month?"
# query_2 = "What are the most traded stock yesterday?"
# query_3 = (
#     "What are the top 7 most traded stocks between 6th June to 10th June this year?"
# )
# query_4 = "What are the top 3 companies by transaction volume over the last 7 days?"  # gak konsisten atara pakai angka dan tidak
# query_5 = "Based on the closing prices of BBCA between 1st and 30th of June 2024, are we seeing an uptrend or downtrend? Try to explain why."
# query_6 = "What is the company with the largest market cap between BBCA and BREN? For said company, retrieve the email, phone number, listing date and website for further research."
# query_7 = "What is the performance of GOTO  since its IPO listing?"
# query_8 = "If i had invested into GOTO vs BREN on their respective IPO listing date, which one would have given me a better return over a 90 day horizon?"
# query_9 = "Give me the 2nd place contact information of the top 3 companies by transaction volume on the 1st of july 2024"
# query_10 = "How much is the average of BBCA close price from 1st of august 2024 until 3 august 2024?"
# query_11 = "Show me the graph of BBRI closing price from 1st of august 2024 until 7 august 2024."
# query_12 = "What is the subsector of BBRI?"
# query_13 = "What is the ABCD.JK overview?"

# queries = [query_13]

# for query in queries:
#     print("Question:", query)
#     x = LLM_Chat()
#     result = x.invoke({"input": query})
#     print("Answer:", "\n", result["output"], "\n\n============================\n\n")

##################################################################

st.title("🚀 Sectors.:red[AI] Chatbot")
st.markdown("Ask me anything about stock market and company information! feat. Sectors")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input():
    if not sectors_api_key or not groq_api_key:
        st.info("Please set your API Keys first before using the chatbot.", icon="🔑")
        st.stop()
    # print(st.session_state)
    st.chat_message("user").markdown(prompt)
    # # check api key
    # if (
    #     st.session_state["sectors_api_key"] == ""
    #     or st.session_state["groq_api_key"] == ""
    # ):
    #     st.warning(
    #         "Please set your API Keys first before using the chatbot.",
    #         icon="🔑",
    #     )
    #     st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.status("🤓 Thonking your question..", expanded=True) as status:
            response = None
            try:
                st_callback = StreamlitCallbackHandler(st.container())
                model = LLM_Chat(st.session_state["groq_api_key"])
                # print("BBBBBBBBBBBBBBBB")
                response = model.invoke(
                    {"input": prompt}, callback_handler=[st_callback]
                )
                # print("CCCCCCCCCCCCCCCCCC")
                # print(response)
                status.update(label="💡 Eureka!", state="complete", expanded=False)
            except Exception as e:
                print(
                    # "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                )
                # status.update(
                #     label=f"🚨 Error, Somethings wrong, reason {type(e).__name__}",
                #     state="error",
                #     expanded=False,
                # )
                # st.stop()
                response = None
                status.update(
                    label=f"❌ {type(e).__name__}", state="error", expanded=False
                )
    if response:
        st.write(response["output"])
        st.session_state.messages.append(
            {"role": "assistant", "content": response["output"]}
        )
    else:
        st.warning(
            f"Something wrong happened. Please try again later.",
            icon="🚨",
        )
        st.stop()

# analyyze the trend of BBRI from 2022 until 2023
# Compare close price of BBRI BMRI, BBCA in august 2024, which one should i buy
