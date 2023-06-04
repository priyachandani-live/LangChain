"""
Description : This file is for writing functions to answer the questions
"""
import os
import json

from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv
import streamlit as st

#Read the file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

#
def csv_tool(filename):
    df = pd.read_csv(filename)
    return create_pandas_dataframe_agent(OpenAI(temperature=0), df=df, verbose=True)

#
def ask_agent(agent, query):
    """
        Query an agent and return the response as a string.
        Args:
            agent: The agent to query.
            query: The query to ask the agent.
        Returns:
            The response from the agent as a string.
    """
    prompt = (
         """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

        1. If the query requires a table, format your answer like this:
           {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        2. For a bar chart, respond like this:
           {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        3. If a line chart is more appropriate, your reply should look like this:
           {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        Note: We only accommodate two types of charts: "bar" and "line".

        4. For a plain question that doesn't need a chart or table, your response should be:
           {"answer": "Your answer goes here"}

        For example:
           {"answer": "The Product with the highest Orders is '15143Exfo'"}

        5. If the answer is not known or available, respond with:
           {"answer": "I do not know."}

        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
        For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

        Now, let's tackle the query step by step. Here's the query for you to work on: 
        """
        + query
    )
    response = agent.run(prompt)
    return str(response)

#
def decode_response(response:str) -> dict:
    return json.loads(response)

#
def write_answer(response_dict:dict):
   """
   Write a response from an agent to a Streamlit app.

   Args:
      response_dict: The response from the agent.

   Returns:
      None.
   """
   # {"answer": "The Product with the highest Orders is '15143Exfo'"}
   if "answer" in response_dict:
      return st.write(response_dict['answer'])
        
   # {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
   if "bar" in response_dict:
      data = response_dict['bar'] #{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}
      try:
         df_data = {
            col : [x[i] if isinstance(x, list) else x for x in data['data']]
            for i , col in enumerate(data['columns'])
         }
         df = pd.DataFrame(df_data)
         df.set_index("Products", inplace=True)
         st.bar_chart(df)
      except ValueError:
         print(f"Couldn't create dataframe from data: {data}")
       
   # {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
   if "line" in response_dict:
      try:
         df_data = {col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])}
         df = pd.DataFrame(df_data)
         df.set_index("Products", inplace=True)
         st.line_chart(df)
      except ValueError:
         print(f"Couldn't create DataFrame from data: {data}")

   if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)


st.set_page_config(page_title="👨‍💻 Talk with your CSV")
st.title("👨‍💻 Talk with your CSV")

st.write("Please upload your CSV file below.")

data = st.file_uploader("Upload a CSV" , type="csv")

query = st.text_area("Send a Message")

if st.button("Submit Query", type="primary"):
   # Create an agent from the CSV file.
   agent = csv_tool(data)

   # Query the agent.
   response = ask_agent(agent=agent, query=query)

   # Decode the response.
   decoded_response = decode_response(response)

   # Write the response to the Streamlit app.
   write_answer(decoded_response)
    
   
