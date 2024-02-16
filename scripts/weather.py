from util import get_coord, WeatherChat, open_ai_key
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
import time
import logging

global messages
messages = [{"role": "Assitant", "content": "What do you want to know about the weather?"}]
def add_messages(message, role) -> None:
    messages.append({"role": role, "content": message})
    return messages

def main():
    

    lat, lon = None, None
    #st.sidebar.title("Choose Model")
    model_choice = input("Select a Model:> \n\nopenai gpt-3.5: 1 \nllama2: 2\n:> ")
    if int(model_choice) == 1:
        open_ai_key()
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    else:
        llm = Ollama(model='llama2')

    msg = messages[0]
    print(f"{msg['role']}: {msg['content']}")
    while True:
        weather_chat = WeatherChat(ip=None, llm=llm)
        if weather_chat:
            chat_agent = weather_chat.initialize()

            user_query = input(f"Ask about the weather")
            base = "Below is the chat history between the User and Assistant\n\n"
            if user_query:
                add_messages(role="User", message= user_query)
                try:
                    
                    all_query = base + '\n'.join([f"{msg['role']}:{msg['content']}" for msg in messages])
                    answer = chat_agent.run(all_query)
                    add_messages(role="Assistant", message= answer)

                    tokens = answer.split(' ')
                    for token in tokens:
                        print(token, end=' ')
                        time.sleep(0.5)

                    #print(answer)
                except ValueError as e:
                    logging.error(msg='Oops! We are sorry, an error occured will generating answer')
main()
                    
