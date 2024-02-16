import streamlit as st
from util import get_coord, WeatherChat, open_ai_key, init_messages, load_openweather, start_chat
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
weather_chat = None
def init_page() -> None:
    st.set_page_config(
    )
    st.sidebar.title("Options")
    icon, title = st.columns([3, 20])
    with icon:
        st.image('./imgs/weatherchat.jpg')
    with title:
        st.title('Weather Chat')

if not st.session_state.get('use_ip'):
    st.session_state['use_ip'] = False

if not st.session_state.get('started'):
    st.session_state['started'] = False


def main():
    init_page()

    

    lat, lon = None, None
    #st.sidebar.title("Choose Model")
    model_choice = st.sidebar.radio("Select a Model", ("openai gpt-3.5", "llama2"))
    if model_choice == "openai gpt-3.5":
        open_ai_key()
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    else:
        llm = Ollama(model='llama2')

    if not st.session_state['use_ip']:
        #consent, user_ip = 
        start_chat()
        st.stop()
        user_ip = st.session_state['userip']
    else:
    #    if st.session_state['use_ip']:
        user_ip = st.session_state['userip']
        weather_chat = WeatherChat(ip=user_ip, llm=llm)

    if weather_chat:
        chat_agent = weather_chat.initialize()
        st.session_state['chat_agent'] = chat_agent
        if not st.session_state['started']:
            init_messages()
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        user_query = st.chat_input(f"Ask about the weather")
        base = "Below is the chat history between the User and Assistant\n\n"
        if user_query:
            st.session_state.messages.append({"role": "User", "content": user_query})
            st.chat_message("user").write(user_query)
            try:
                with st.spinner('Generating'):
                    all_query = base + '\n'.join([f"{msg['role']}:{msg['content']}" for msg in st.session_state.messages])
                    answer = chat_agent.run(all_query)
                    st.session_state.messages.append({"role": "Assistant", "content": answer})

                st.write(answer)
            except ValueError as e:
                st.error('Oops! We are sorry, an error occured will generating answer')
    else:
        openweather_agent = load_openweather(llm)
        if openweather_agent:
            init_messages()
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

            user_query = st.chat_input("What is the weather like in Abuja?")
            if user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                st.chat_message("user").write(user_query)
                try:
                    with st.spinner('Generating'):
                        answer = openweather_agent.run(user_query)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.write(answer)
                except ValueError as e:
                    st.error('Oops! We are sorry, an error occured will generating answer')
        st.warning("Location data not provided.")


main()
                    
