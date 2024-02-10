import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry
from typing import Optional, Type

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
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

from langchain.agents import AgentType, initialize_agent, load_tools

#os.environ["OPENWEATHERMAP_API_KEY"] = "47fe1d37e973b177509b6441d93bc582"



def open_ai_key():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
    os.environ["OPENAI_API_KEY"] = openai_api_key


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
class WeatherChat:
    def __init__(self,lat, lon, llm=None, city=None, country=None):
        
        self.city = city
        self.country = country
        self.__name__ = f"User@({lon},{lat})"
        self.llm = llm
        
        self.lat = lat
        self.lon = lon
        
        self.memory = ConversationBufferMemory(key='memory')
    
        
    def __get_weather_update(self):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "is_day", "precipitation", "rain", "showers", "snowfall", "weather_code", "cloud_cover", "pressure_msl", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "precipitation", "rain", "showers", "snowfall", "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "visibility", "evapotranspiration", "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m", "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m", "wind_gusts_10m", "temperature_80m", "temperature_120m", "temperature_180m", "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"],
            "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min", "sunrise", "sunset", "daylight_duration", "sunshine_duration", "uv_index_max", "uv_index_clear_sky_max", "precipitation_sum", "rain_sum", "showers_sum", "snowfall_sum", "precipitation_hours", "precipitation_probability_max", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        self.response = response
        return response
    
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
    
    def __extract_hourly(self):
        hourly = self.response.Hourly()

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

        hourly_data = {"time": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s"),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        hourly_data.update(hourly_dict)


        hourly_dataframe = pd.DataFrame(data = hourly_data)
        hourly_dataframe['time'] = hourly_dataframe['time'].apply(lambda val: val.time())
        return hourly_dataframe
    
    def __extract_daily(self):
    
        daily = self.response.Daily()

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
    
    def initialize(self):
        self.__get_weather_update()
        current_data = self.__extract_current()
        daily_data  = self.__extract_daily()
        hourly_data = self.__extract_hourly()
        
        def run_current(query:str)-> str:
            
            """ Use this tool when you need to look up on daily weather forecast"""

            print("Entering New Current Weather Chain")
            print("This is the prompt gotten", query)
            now = datetime.now()
            
            daily_prompt = f"""You have been provided with current weather data  for {self.city}, {self.country} as a JSON below to answer your questions.\n"""
            if query is None:
                query = " "
            prompt = daily_prompt + f"Current Weather Data: {current_data}\n\n Summarise the data to answer the specific question below"+ query + "If no specific question then you should give an overview of the weather information"
            
            res = self.llm.predict(prompt)
            #agent = create_pandas_dataframe_agent(
            #self.llm,
            #current_data,
            #verbose=True,
            #agent_type=AgentType.OPENAI_FUNCTIONS
            #)

            #res = agent.run(prompt)
            self.tool_response = res
            return res
        
        def run_daily(query:str)-> str:
            
            """ Use this tool when you need to look up on daily weather forecast"""

            print("Entering New Daily Weather Chain")
            print("This is the prompt gotten", query)
            now = datetime.now()
            current_day = now.date().strftime("%d %B, %Y")
            daily_prompt = f"""Today is {current_day}. You have been provided with weather data for {self.city}, {self.country} to answer your questions.\n"""
            prompt = daily_prompt + "Give the weather information on this "+ query

            agent = create_pandas_dataframe_agent(
            self.llm,
            daily_data,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
            )

            res = agent.run(prompt)
            self.tool_response = res
            return res


        def run_hourly(query):
            
            """ Use this tool when you need to look up on hourly weather forecast"""

            print("Entering New Hourly Weather Chain")
            print("This is the prompt gotten", query)
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            hourly_prompt = f"""The current time is {current_time}. You have been provided with hourly weather data for {self.city}, {self.country} to answer your questions"""
            print(type(query))
            prompt = hourly_prompt + "Give the weather information on this "+ query

            agent = create_pandas_dataframe_agent(
            self.llm,
            hourly_data,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
            )

            res = agent.run(prompt)
            self.tool_response = res
            return res
        
        def chat_openweather(query):
            weather = OpenWeatherMapAPIWrapper()
            weather_data = weather.run(query)
            return weather_data



        
        tools = [
            Tool.from_function(run_current,
                               name = "current_weather",
                               description= "Use this tool when you need to answer questions on the current weather "
                              ), 
            Tool.from_function(run_daily,
                               name = "daily_weather",
                               description= "Use this tool when you need to answer questions on daily weather forecast"
                              ), 

            Tool.from_function(run_hourly,
                              name= "Hourly_weather",
                              description = "Use this when you need to answer questions on hourly weather forecast"),
            ]


        agent = initialize_agent(
        tools = tools,
        llm=self.llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory = self.memory
        )
        self.agent = agent
        return self
    
    def run(self, query):
        try:
            res = self.agent.run(query)
            return f"In {self.city}, {self.country}.\n\n"+ res
        except ValueError as e:
            return self.tool_response
        

def load_openweather(llm):
    weather = OpenWeatherMapAPIWrapper(openweathermap_api_key="47fe1d37e973b177509b6441d93bc582")
    os.environ["OPENWEATHERMAP_API_KEY"] ="47fe1d37e973b177509b6441d93bc582"
    tools = load_tools(["openweathermap-api"], llm)

    agent_chain = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    
    return agent_chain

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?", "img_path": None}]
    
    