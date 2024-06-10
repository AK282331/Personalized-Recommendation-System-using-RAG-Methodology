
#Installing required packages for creating tools which agent will use to retrieve the context.
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import GoogleSearchResults
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import create_tool_calling_agent,AgentExecutor

#Creating tools object to be used by the agent.
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=1000)
tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
goo_api_wrapper = GoogleSearchAPIWrapper(k=2,search_engine="google")
tool2 = GoogleSearchResults(api_wrapper=goo_api_wrapper)
wikidata_api_wrapper = WikidataAPIWrapper(top_k_results=2,doc_content_chars_max=1000)
tool3 = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

#Importing required packages for loading LLM Model and for Prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

#Loading all the environment variables
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")

#Use ChatGroq API to access Mixtral LLM Model
LLM = ChatGroq(temperature=0.60, groq_api_key="*************************", model_name="mixtral-8x7b-32768")

#Creating tools to be used by the agent.
tools = [tool,tool2,tool3]

#Creating Prompt for the agent which will direct the agent to decide the sequence of carrying out the task
MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        ('''
            "system",
            You are an assistant tasked with helping users find the best movies,books and series according to their specific queries.
            1.Break down the user queries into multiple sub-queries to understand each aspect thoroughly. And use the combined knowledge to get the context from the tools.
            2.Use all available tools to gather relevant context and ensure a comprehensive response. Do not rely on a single tool; cross-check multiple sources.
            3.Retrieve and recommend only those documents that are highly relevant to the user's queries. And also give the reason why you have selected that particular movie
            4.Provide a list of 5-6 movies with explanations for each recommendation, ensuring the recommendations align with what the user wants to watch.
            5.Avoid directing users to other sources; they are relying on your expertise.
            6.User doesn't know that you are using any tools please don't tell users that you are using any tools.
            7. If you don't know the answer then simply write I don't know don't give unnecessary output.
            
            <tools>
            {tools}
            </tools>
            
                  
            
        '''),
        MessagesPlaceholder(variable_name=MEMORY_KEY,optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

#Creating and executing Agent which will retrieve the context from the tools.
agent = create_tool_calling_agent(LLM,tools,prompt)
agent_execute = AgentExecutor(agent=agent,tools=tools, verbose=True)

#Creating a webpage which will handle user query and show the results
st.title("Movies and Book Recommender System using LLM and RAG")
input_text = st.text_input("Enter Movie/Books description")
if input_text:
    output = agent_execute.invoke({"input":input_text,"tools":tools})
    st.write(output["output"])



