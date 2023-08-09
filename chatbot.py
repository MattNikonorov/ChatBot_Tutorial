from flask import Flask, request, jsonify
from flask import Flask, render_template, request, url_for

from llama_index import SimpleDirectoryReader, GPTListIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import time
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas
import openai
import numpy as np
import glob
import datetime

os.environ["OPENAI_API_KEY"] = 'YOUR_KEY'

openai.api_key = 'YOUR_KEY'

ips = []
ips_times = []

ips_ref = []
ips_times_ref = []

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate


llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']



def logic(question):
    
    df = pandas.read_csv(f"embs0.csv")

    embs = []
    for r1 in range(len(df.embedding)): # Changing the format of the embeddings into a list due to a parsing error
        e1 = df.embedding[r1].split(",")
        for ei2 in range(len(e1)):
            e1[ei2] = float(e1[ei2].strip().replace("[", "").replace("]", ""))
        embs.append(e1)

    df["embedding"] = embs

    bot_message = ""
    product_embedding = get_embedding( # Creating an embedding for the question that's been asked
        question
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding)) # Finds the relevance of each piece of data in context of the question
    df.to_csv("embs0.csv")

    df2 = df.sort_values("similarity", ascending=False) # Sorts the text chunks based on how relevant they are to finding the answer to the question
    df2.to_csv("embs0.csv")
    df2 = pandas.read_csv("embs0.csv")
    #print(df2["similarity"][0])

    from langchain.docstore.document import Document

    comb = [df2["combined"][0]]
    docs = [Document(page_content=t) for t in comb] # Gets the most relevant text chunk

    prompt_template = question + """

    {text}

    """ 

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT) # Preparing the LLM

    output = chain.run(docs) # Formulating an answer (this is where the magic happens)

    return output



response = logic("when was the first computer made?") # Passing the question to the ChatBot

print(response)
