from agent_graph import OurLangchainCallbackHandler, create_agent_graph, llm_options, prompt_engineer
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
from zep_cloud.client import Zep
from zep_cloud.errors import NotFoundError
import panel as pn
import uuid

# ---------------------------------------------------------------------------- #
#                                  Panel Setup                                 #
# ---------------------------------------------------------------------------- #
CORA = "Cora"
accent_color = "#A01346"
pn.extension(notifications=True, loading_indicator=True, global_loading_spinner=True)

# ---------------------------------------------------------------------------- #
#                                  Zep Setup                                   #
# ---------------------------------------------------------------------------- #

zep_client = Zep()


def setup_user(email):
    logger.info(f"Setting up user {email}")
    # Extract username from email
    username = email.split("@")[0] if "@" in email else email

    # Try to get the user
    try:
        zep_user = zep_client.user.get(email)
    except NotFoundError:
        logger.info(f"User {email} not found, creating new user")
        zep_user = zep_client.user.add(user_id=email, email=email)

    # Create avatar
    avatar = username[0].upper()

    return zep_user, username, avatar


# Set up user and avatar
zep_user, username, avatar = setup_user(pn.state.user)

# ------------------------------ Settings Modal ------------------------------ #

# Define the settings
settings = pn.state.as_cached(
    "settings",
    lambda: {
        "llm": llm_options[0],
        "search_web": True,
        "use_prompt_engineering": True,
    },
)

# Create widgets for settings
llm_select = pn.widgets.Select(name="Language Model", options=llm_options, value=settings["llm"])
search_web_checkbox = pn.widgets.Checkbox(name="Search the Web (when needed)", value=settings["search_web"])
use_prompt_engineering_checkbox = pn.widgets.Checkbox(
    name="Use Prompt Engineering", value=settings["use_prompt_engineering"]
)


# Function to update settings
def update_settings(event):
    settings["llm"] = llm_select.value
    settings["search_web"] = search_web_checkbox.value
    settings["use_prompt_engineering"] = use_prompt_engineering_checkbox.value
    pn.state.notifications.success("Settings updated", duration=2500)


# Create settings modal
settings_modal = pn.Column(
    "## Settings",
    llm_select,
    search_web_checkbox,
    use_prompt_engineering_checkbox,
    pn.widgets.Button(name="Save", button_type="primary", on_click=update_settings),
)

# Create settings button
settings_button = pn.widgets.Button(name="⚙️ Settings", button_type="default")

# ---------------------------------------------------------------------------- #
#                               LLM Interactions                               #
# ---------------------------------------------------------------------------- #

chat_history = []
thread_id = str(uuid.uuid4())  # Generate a unique thread ID


def callback(user_request, user, chat_interface: pn.chat.ChatInterface):
    agent_graph = create_agent_graph(settings)

    # Pre-process the user request if prompt engineering is enabled
    if settings["use_prompt_engineering"]:
        enhanced_request = prompt_engineer(user_request).content
        chat_interface.send(
            f"*Enhanced Request*:\n\n{enhanced_request}", user="Prompt Engineer", avatar="🎛", respond=False
        )
    else:
        enhanced_request = user_request

    # Set up the agent and the callback handler
    callback_handler = OurLangchainCallbackHandler(chat_interface, user=CORA, avatar="assets/logo.png")
    runnable_config = RunnableConfig(configurable={"thread_id": thread_id})
    runnable_config["callbacks"] = [callback_handler]

    chat_history.append(HumanMessage(content=enhanced_request))
    result = agent_graph.invoke({"messages": chat_history}, config=runnable_config)

    ai_response = result["messages"][-1].content
    if isinstance(ai_response, list):
        ai_response_text = ai_response[0]["text"]
    else:
        ai_response_text = ai_response

    chat_history.append(AIMessage(content=ai_response_text))


# ---------------------------------------------------------------------------- #
#                          Panel Interface and Layout                          #
# ---------------------------------------------------------------------------- #


# Create the ChatInterface with our custom options
chat_interface = pn.chat.ChatInterface(
    user=username,
    avatar=avatar,
    callback=callback,
    callback_exception="verbose",
    callback_user=CORA,
    show_rerun=False,
    show_stop=False,
    show_undo=False,
    message_params={
        "show_timestamp": False,
        "reaction_icons": {},
        "default_avatars": {"Assistant": "assets/logo.png"},
    },
)

template = pn.template.FastListTemplate(
    title="Cora: Heart-Centered AI 🤖 + 💙",
    header=[settings_button],
    main=[chat_interface],
    logo="assets/logo.png",
    favicon="assets/logo.png",
    corner_radius=5,
    theme_toggle=False,
    accent=accent_color,
)

# Use Panel's built-in modal functionality
template.modal.append(settings_modal)

# Open the modal when the settings button is clicked
settings_button.on_click(lambda event: template.open_modal())

template.servable()
