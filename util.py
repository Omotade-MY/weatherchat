import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry
from typing import Optional, Type
import threading
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

import openai
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.agents import tools, Tool, tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from datetime import datetime
import os


import streamlit as st
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.agents.react.agent import create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import AgentType, initialize_agent, load_tools
import numpy as np
os.environ["OPENWEATHERMAP_API_KEY"] = "47fe1d37e973b177509b6441d93bc582"



def open_ai_key():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
    os.environ["OPENAI_API_KEY"] = openai_api_key

def randomName():
    n = []
    n.append(np.random.choice(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']).upper())
    n.append(str(np.random.randint(1,9)))                                 
    n.append(str(np.random.randint(1,9)))   
    n.append(np.random.choice(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']))         
    n.append(np.random.choice(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']).upper())  

    return ''.join(n)       

def get_coord(ip):
    import requests

    try:
        url = f"https://ipinfo.io/{ip}"
        res = requests.get(url).json()

        res_dict = {'city':res['city'], 'region':res['region'], 'country':res['country'],
                'lat':float(res['loc'].split(',')[0]), 'lon':float(res['loc'].split(',')[1])}
        
        return res_dict
    except Exception as e:
        return None
from typing import Optional, Type
from typing import Optional, Type
class WeatherChat:
    def __init__(self, llm,ip=None):
        
        #self.city = city
        self.country = None
        self.__name__ = f"User@{ip}"
        self.openai_key =  "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"
        self.llm = llm
        self.ip = ip
        #self.lat = lat
        #self.lon = lon
        self.base_prompt = None
        # initiatiate memory
        self.memory = ConversationBufferMemory(key='memory')
    
        
    def __get_weather_update_daily(self,lat, lon):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
           "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min", "sunrise", "sunset", "daylight_duration", "sunshine_duration", "uv_index_max", "uv_index_clear_sky_max", "precipitation_sum", "rain_sum", "showers_sum", "snowfall_sum", "precipitation_hours", "precipitation_probability_max", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        #self.response = response
        #return response
        daily = response.Daily()

        daily_features = ["weather_code", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", 
                          "apparent_temperature_min", "sunrise", "sunset", "daylight_duration", "sunshine_duration", 
                          "uv_index_max", "uv_index_clear_sky_max", "precipitation_sum", "rain_sum", "showers_sum", 
                          "snowfall_sum", "precipitation_hours", "precipitation_probability_max", "wind_speed_10m_max", 
                          "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", 
                          "et0_fao_evapotranspiration"]
        

        daily_dict = {f'{feat}':daily.Variables(i).ValuesAsNumpy() for i, feat in enumerate(daily_features)}

        daily_data = {"date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s"),
            end = pd.to_datetime(daily.TimeEnd(), unit = "s"),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        )}
        daily_data.update(daily_dict)


        daily_dataframe = pd.DataFrame(data = daily_data)
        
        return daily_dataframe
    
    def __extract_current(self):
        current = self.response.Current()
        current_features = ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "is_day", "precipitation", "rain", "showers", "snowfall", "weather_code", "cloud_cover", "pressure_msl", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]
        current_dict = {f'current_{feat}':current.Variables(i).Value() for i, feat in enumerate(current_features)}

        current_dict['Time'] = current.Time()

        current_df = pd.Series(current_dict).to_frame()
        current_df.reset_index(inplace=True)
        current_df.columns = ['Weather ', 'Value'] 
        #return current_df
        return current_dict
    
    def __get_weather_update_hourly(self,lat, lon):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
           "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "precipitation", "rain", "showers", "snowfall", "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "visibility", "evapotranspiration", "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m", "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m", "wind_gusts_10m", "temperature_80m", "temperature_120m", "temperature_180m", "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"],
                 }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        #self.response = response
        #return response


        hourly = response.Hourly()

        hourly_features = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", 
                           "precipitation_probability", "precipitation", "rain", "showers", "snowfall", "snow_depth", 
                           "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", 
                           "cloud_cover_high", "visibility", "evapotranspiration", "et0_fao_evapotranspiration", 
                           "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m", 
                           "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m", 
                           "wind_gusts_10m", "temperature_80m", "temperature_120m", "temperature_180m", "soil_temperature_0cm", 
                           "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm", 
                           "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"]

        hourly_dict = {f'{feat}':hourly.Variables(i).ValuesAsNumpy() for i, feat in enumerate(hourly_features)}

        hourly_data = {"date_time": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s"),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}

        hourly_data.update(hourly_dict)


        hourly_dataframe = pd.DataFrame(data = hourly_data)
        hourly_dataframe['time'] = hourly_dataframe['date_time'].apply(lambda val: val.time())
        return hourly_dataframe
    
    def initialize(self):
        
        def extract_addr(query):
            base = """Extract the names of cities, regions, or countries from the provided text. Separate each named entity by a comma.\n\n
                if no named entities respond strictly with "NO Location Found"
                Text: {}
                """

            prompt = base.format(query)
            res = llm.invoke(input=prompt)
            return res.content



        def get_coord2(city):
            import requests
            url = f"https://geocode.maps.co/search?q={city}&api_key=65c4b7ee942df610164192nzce4ad7e"
            result = requests.get(url).json()
            res = result[0]

            return float(res['lat']), float(res['lon'])
        

        def get_coord(ip=None):
            ip = self.ip
            try:
                assert ip is not None
            except AssertionError:
                st.warning('No IP Found. Permision to use location was not granted')
                return "No IP Found. The user has not granted permision to use their location"
            #permission = input()
            
            import requests

            try:
                url = f"https://ipinfo.io/{self.ip}"
                res = requests.get(url).json()

                res_dict = {'city':res['city'], 'region':res['region'], 'country':res['country'],
                        'lat':float(res['loc'].split(',')[0]), 'lon':float(res['loc'].split(',')[1])}



                return res_dict
            except Exception as e:
                return None
        def run_daily(query:str)-> str:

            """ Use this tool when you need to look up on daily weather forecast"""

            print("Entering New Daily Weather Chain")
            print("This is the prompt gotten", query)
            #if self.ip is None:
            city = extract_addr(query)
            print("City: ", city)
            lat, lon = get_coord2(city)
            #else:
            #    locs = get_coord(ip=self.ip)
            #    city = ', '.join([locs['city'],locs['country']])
            #    lat, lon = float(locs['lat']), locs['lon']
            print('Lat and Lon: ({}, {})'.format(lat,lon))
            
            daily_data = self.__get_weather_update_daily(lat, lon)
            
            print("This is daily weather date: ", daily_data['date'])
            
            now = datetime.now()
            current_day = now.date().strftime("%d %B, %Y")
            daily_prompt = f"""Given today's date, {current_day}, you have weather forecast data available for {city} to address any inquiries. Ensure you always provide a total and conclusive weather information from the data you have.\n\n"""
            prompt = daily_prompt + query

            agent = create_pandas_dataframe_agent(
            llm,
            daily_data,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
            )

            res = agent.run(prompt)
            return res


        def run_hourly(query:str)-> str:

            """ Use this tool when you need to look up on daily weather forecast"""

            print("Entering New Hourly Weather Chain")
            print("This is the prompt gotten", query)
            
            #if self.ip is None:
             
            city = extract_addr(query)
            print("City: ", city)
            lat, lon = get_coord2(city)
            
            #else:
            #    locs = get_coord(ip=self.ip)
            #    city = ', '.join([locs['city'],locs['country']])
            #    lat, lon = float(locs['lat']), locs['lon']
            print('Lat and Lon: ({}, {})'.format(lat,lon))
            hourly_data = self.__get_weather_update_hourly(lat, lon)
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            hourly_prompt = f"""The current time is {current_time}. You have been provided with hourly weather data for {city} to address any inquiries. You always expected to provide a conclusive weather information from the data you have\n\n"""
            print(type(query))
            prompt = hourly_prompt + query

            agent = create_pandas_dataframe_agent(
            llm,
            hourly_data,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
            )
            
            res = agent.run(prompt)
            return res
            

        def chat_openweather(loc):
    
            if loc.lower() in ['current_location', 'current location']:
                locs = get_coord()
            
                loc = ', '.join([locs['city'],locs['country']])
            #elif loc.split()
            
            

            weather = OpenWeatherMapAPIWrapper()
            try:
                
                weather_data = weather.run(loc)
                print(loc)
                return weather_data
            
            except Exception as e:
                print(str(e))
                loc = get_coord()
                loc = ', '.join([locs['city'],locs['country']])
            
            


        def greet(greeting):
            #llm = self.llm #ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613", openai_api_key = "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")
            #res = llm.invoke(greeting)
            #return res.content
            return "Hello!, This is Weather Chat, How may I help you with your weather questions?"



        
        tools = [
                Tool.from_function(chat_openweather,
                                  name= "chat_open_weather",
                                  description = "Use this when you need to provide current weather information on a specified loaction. The input to this tool is can only be a specific location e.g city, country, region, or the word 'current location'"),
                Tool.from_function(
                                greet,
                                name='greet',
                                description = "Use this too only  for responding to greetings. This tool does not require an input"
                ),
                Tool.from_function(
                                get_coord,
                                name="current_location",
                                description = "useful if you need to know the current location. This tool does not take any input"
                ),
                Tool.from_function(
                                run_daily,
                                name= 'daily_weather_forecast',
                                description = "Useful for getting daily weather forecast information for any location. The input to this function is the full prompt message. Always include the location"
                ),
                Tool.from_function(
                                run_hourly,
                                name= 'hourly_weather_forecast',
                                description = "Useful for getting hourly weather forecast information for any location. TThe input to this function is the full prompt message. Always include the location"
                )
            ]

        # Choose the LLM to use
        # Get the prompt to use - you can modify this!
        prompt = hub.pull("hwchase17/react")
        if self.base_prompt is None:
            self.base_prompt = prompt.template
        prompt.template = """As an expert meteorologist, you'll engage in detailed and accurate weather discussions with a human. You must always detemine the location for the weather analysis. The location may be provided directly or implied in the context. In cases where it's unclear, assume the user refers to their current location. However if you are greeted then you simply respond with a greeting.\n\n""" + self.base_prompt
        llm = self.llm #ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613", openai_api_key = "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")

        # Construct the ReAct agent
        agent = create_react_agent(llm, tools, prompt)

        # Create an agent executor by passing in the agent and tools
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        #agent_executor.invoke({'input': "what is the weather in Abuja today"})
        self.agent = agent_executor
        return self
    
    def run(self, query):
        base = ""
        res = self.agent.invoke({'input':query})
        self.response = res
        return res['output']
    
    
    
    
        

def load_openweather(llm):
    weather = OpenWeatherMapAPIWrapper(openweathermap_api_key="47fe1d37e973b177509b6441d93bc582")
    os.environ["OPENWEATHERMAP_API_KEY"] ="47fe1d37e973b177509b6441d93bc582"
    tools = load_tools(["openweathermap-api"], llm)

    agent_chain = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    
    return agent_chain
import ipaddress

def is_valid_ip(ip_address):
    try:
        # Attempt to create an IPv4 or IPv6 address object
        ipaddress.ip_address(ip_address)
        # If successful, return True
        return True
    except ValueError:
        # If an error is raised (invalid IP address), return False
        return False

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    
    if clear_button or "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", 
                                         "content": "Welcome to WeatherChat!! What do you want to know about the weather?"}]
