import streamlit as st
from util import get_coord, WeatherChat, open_ai_key, init_messages, load_openweather, is_valid_ip
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
import openai


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
    st.session_state['use_ip'] = None
#st.session_state['use_ip'] = None

if not st.session_state.get('userip'):
    st.session_state['userip'] = None

if not st.session_state.get('started'):
    st.session_state['started'] = False

#st.session_state['started'] = False

def start_chat() -> None:
    print("Entered Start Cart")
    if (st.session_state['use_ip'] is None) or (st.session_state['userip'] is None):
        st.session_state["messages"] = [{"role": "assistant", 
                                            "content": "Welcome to WeatherChat!! We need your permission to use your IP address to provide weather information for your current location. May we proceed? .\n Answer with YES or NO"}]
        st.chat_message("assistant").write("Welcome to WeatherChat!! We need your permission to use your IP address to provide weather information for your current location. May we proceed?")
    
    consent = st.chat_input("Enter 'yes' or 'no'")
    
    if consent:
        st.session_state.messages.append({"role": "user", "content":consent})
            
        if consent.strip().lower() not in ("yes", "no"):
            st.chat_message('assistant').write("Please respond only with 'yes' or 'no'")
            st.session_state.messages.append({"role": "assistant", "content": "Please respond only with 'yes' or 'no'"})
            

        if consent.strip().lower() == "yes":
            st.session_state['use_ip'] = True
            
        elif consent.strip().lower() == "no":
            st.session_state['use_ip'] = False
            st.session_state['started'] = True
            st.session_state["messages"] = []
            st.rerun()

        else:
            return

    if (st.session_state['use_ip']) and (not st.session_state['userip']):
        print("WE GOT HERE")
        st.chat_message('assistant').write("Please provide your IP address to continue")
        st.session_state.messages.append({"role": "assistant", "content": "Please provide your IP address to continue"})
        user_ip = st.chat_input("Enter your IP address e.g 192.168.1.1 ")

        print(user_ip)
        print(type(user_ip))

        if user_ip :
            print(user_ip)
            st.chat_message('user').write(user_ip)
            st.session_state.messages.append({"role": "user", "content": user_ip})
            
            if is_valid_ip(user_ip.strip()):
                if st.session_state['userip'] is None:
                    st.session_state['userip'] = user_ip.strip()
                st.session_state['started'] = True
                st.session_state["messages"] = [] 
                st.rerun()

            else:
                st.chat_message('assistant').write("The IP address you have provided is not valid. Please provide a valid IP address to continue")
                st.session_state.messages.append({"role": "assistant", "content": "The IP address you have provided is not valid. Please provide a valid IP address to continue"})

  
        
    

def main():
    weather_chat = None
    init_page()

    model_choice = st.sidebar.radio("Select a Model", ("openai gpt-3.5", "llama2"))
    if model_choice == "openai gpt-3.5":
        open_ai_key()
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    else:
        llm = Ollama(model='llama2')

    #print(st.session_state['started'])
    
    if (not st.session_state['started']): 
        start_chat()
    else:
        if st.session_state['use_ip']:
            userip = st.session_state['userip']
            try:
                assert userip is not None
                print(st.session_state['started'])
                weather_chat = WeatherChat(ip=userip, llm=llm)
            except AssertionError:
                st.warning('IP Address not Found')
                st.stop()
        #    user_ip = st.session_state['userip']
        else:
            print(st.session_state['started'])
            weather_chat = WeatherChat(ip=None, llm=llm)

        if weather_chat:
            chat_agent = weather_chat.initialize()
            st.session_state['chat_agent'] = chat_agent
            
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
                
                except openai.RateLimitError as rate_err:
                    err = str(rate_err)
                    st.warning(err)
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
                    
