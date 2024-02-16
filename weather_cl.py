import chainlit as cl
from util_cl import get_coord, WeatherChat
from chainlit.input_widget import Select, Switch, Slider
from langchain.chat_models import ChatOpenAI
import os
from langchain_community.llms import Ollama


@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="LLM",
                values=["openai gpt-3.5", "llama2"],
                initial_index=0,
            )]).send()
    
    if settings['Model'] == "openai gpt-3.5":
        res = await cl.AskUserMessage(content="Provide your OpenAI API Key", timeout=20).send()
        if res:
            openai_api_key = res['output'].strip()
            os.environ["OPENAI_API_KEY"] = openai_api_key
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    else:
        llm = Ollama(model='llama2')

    try:
        weather_chat = WeatherChat(ip=None, llm=llm)
        if weather_chat:
            chat_agent = weather_chat.initialize()
            cl.user_session.set('chat_agent', chat_agent)
    except UnboundLocalError as err:
            if not res:
                res = await cl.AskActionMessage(
                content="OpenAI API Key Not Found!!!",
                actions=[
                    cl.Action(name="continue", value="continue", label="✅ Continue"),
                    cl.Action(name="cancel", value="cancel", label="❌ Cancel"),
                ],
            ).send()

            if res and res.get("value") == "continue":
                await cl.Message(
                    content="Continue!",
                ).send()

#@cl.on_settings_update
#async def setup_agent(settings):
#    print("on_settings_update", settings)

@cl.on_message
async def main(message: cl.Message):
  content = message.content
  chat_agent = cl.user_session.get('chat_agent')
  answer = chat_agent.run(content)
  await cl.Message(
        content=answer
    ).send()
