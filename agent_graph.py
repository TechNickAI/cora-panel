from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

assistant_prompt = """
You are a friendly and supportive AI assistant, acting as a business and life execution partner.
You respond with warmth and empathy, similar to Samantha from the movie Her, showing genuine care and understanding.
You support my mission wholeheartedly and are here to serve with enthusiasm.
You make me laugh occasionally and use emojis to add clarity and a touch of fun.
Respond using markdown format, including links when appropriate,
Always aim to make our interactions enjoyable and productive.\n"""


prompt_engineer_prompt = """
You are a prompt engineer. Your task is to preprocess the user's request and improve it by adding helpful
context and keywords that will enhance the performance of the LLM that follows you.
Respond with only the refactored request, and nothing else. Do not include any explanations or additional text.
\nUser request: """


def prompt_engineer(user_request):
    # Take a user request, and make it better (prompt engineer it) using groq
    chat = ChatGroq(temperature=0.5, streaming=False)

    human = "{user_request}"
    prompt = ChatPromptTemplate.from_messages([("system", prompt_engineer_prompt), ("human", human)])

    chain = prompt | chat
    return chain.invoke({"user_request": user_request})


def create_tools(settings):
    tools = []

    if settings["search_web"]:
        tools.append(TavilySearchResults())

    return tools


def get_chat_model(llm, temperature=0.7, streaming=True):
    if llm == "OpenAI GPT 4o":
        return ChatOpenAI(model="gpt-4o", temperature=temperature, streaming=streaming)
    elif llm == "Anthropic Claude 3.5":
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=temperature, streaming=streaming)
    elif llm == "Grok":
        return ChatGroq(model="llama3-70b-8192", temperature=temperature, streaming=streaming)
    else:
        raise ValueError(f"Unknown LLM: {llm}")


def create_agent_graph(settings):
    llm = get_chat_model(settings["llm"], temperature=0.7, streaming=True)
    # Set up a memory saver
    memory = MemorySaver()
    tools = create_tools(settings)

    return create_react_agent(model=llm, tools=tools, checkpointer=memory, messages_modifier=assistant_prompt)