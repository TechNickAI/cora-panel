import panel as pn
from agent_graph import create_agent_graph

from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

pn.extension()

agent_graph = create_agent_graph({"llm": "Anthropic Claude 3.5", "search_web": True})

chat_history = []

from rich import inspect


def callback(contents, user, instance):
    callback_handler = pn.chat.langchain.PanelCallbackHandler(instance)
    runnable_config = RunnableConfig(configurable={"thread_id": 1})
    runnable_config["callbacks"] = [callback_handler]

    chat_history.append(HumanMessage(content=contents))

    result = agent_graph.invoke({"messages": chat_history}, config=runnable_config)

    inspect(result)

    ai_response_text = result["messages"][-1].content[0]["text"]

    chat_history.append(AIMessage(content=ai_response_text))

    return ai_response_text


pn.chat.ChatInterface(callback=callback, callback_exception="verbose").servable()
