from agent_graph import create_agent_graph, prompt_engineer
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
import panel as pn

pn.extension(notifications=True)

agent_graph = create_agent_graph({"llm": "Anthropic Claude 3.5", "search_web": True})

chat_history = []


def callback(user_request, user, chat_interface: pn.chat.ChatInterface):
    logger.debug(chat_history)

    # Pre-processing
    enhanced_request = prompt_engineer(user_request).content
    chat_interface.send(f"*Enhanced Request*:\n\n{enhanced_request}", user="System", respond=False)

    # Set up the agent and the callback handler
    callback_handler = pn.chat.langchain.PanelCallbackHandler(chat_interface)
    runnable_config = RunnableConfig(configurable={"thread_id": 1})
    runnable_config["callbacks"] = [callback_handler]

    chat_history.append(HumanMessage(content=enhanced_request))
    result = agent_graph.invoke({"messages": chat_history}, config=runnable_config)

    ai_response_text = result["messages"][-1].content[0]["text"]
    chat_history.append(AIMessage(content=ai_response_text))

    return ai_response_text


pn.chat.ChatInterface(callback=callback, callback_exception="verbose").servable()
