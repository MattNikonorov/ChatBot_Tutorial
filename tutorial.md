# How to create a ChatBot Web App and train it on your own data using the OpenAI API

In this article, you'll learn how to train and test your own ChatBot as well as turn it into a web app using the OpenAI API.

## Table of Contents
- Why make a ChatBot?
- Getting started with the OpenAI API
- Preparing your data
- Training and testing a simple ChatBot on your data
- Perfecting your ChatBot
- Turning your ChatBot into a web app

## Why make a ChatBot?
With AI having revolutionised the IT landscape in 2023, many have leveraged this movement using API providers such as OpenAI to integrate AI into their data.

A particularly good way of using AI for your ideas is making your own ChatBot.

For example, if you have a dataset of thousands of company earnings reports and you would like to explore and analyse it without getting old, a good idea would be to make a ChatBot to answer any questions you may have on the documents that you would have to manually search for otherwise. For example, you may want to ask "What year did tech companies have the best earnings", a question that you would usually have to answer by manually digging around your dataset. Luckily using a ChatBot trained on your data, you can get the answer to that question in a matter of seconds.

## Getting started with the OpenAI API
To get started on your very own ChatBot, your first need access OpenAI API. To get your OpenAI API, sign up on their [website](https://openai.com/), then click your profile icon located at the top-right corner of the home page, select "View API Keys." and click "Create New Secret Key" to generate a new API key.

## Downloading a Dataset
For this tutorial I'll be using the Wikipedia page for computers to make a simple ChatBot that can answer any general question about computers and their history.
You can download the dataset in text format [here](https://github.com/MattNikonorov/ChatBot_Tutorial/blob/main/Computer.txt).

## Training your ChatBot
Once you have your API key and dataset file, you can get started with the actual code.
Go to your favourite text-editor, create a new folder where you'll be making your ChatBot, and create an empty python file inside your new project folder.
Make sure to also create a folder named "chatbot_docs" inside your project folder and paste the dataset file into that folder (the name of the folder doesn't matter but for this tutorial it's much easier to name it "chatbot_docs").

Once you've done that, download the libraries that we're going to be using by running the following in your terminal.
```Bash
pip3 install langchain flask llama_index gradio openai pandas numpy glob datetime
```

Finally, once you've installed all the necessary libraries, paste the following code into your python file.
```Python
# Importing the libraries (you won't need all of them right now but you will need them later)

  

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

  
  

openai.api_key =  'YOUR_KEY'  # Setting your API key

  

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") # Setting your OpenAI model

  

gfiles = glob.glob("chatbot_docs/*") # Reading your document directory

  

for g1 in  range(len(gfiles)): # Iterating through every document

  

f =  open(f"embs{g1}.csv", "w") # Creating a csv file for storing the embeddings for your ChatBot

f.write("combined") # Creating the 'combined' collumn

f.close()

  

content =  ""

with  open(f"{gfiles[g1]}", 'r') as file: # Storing the document contents

content += file.read()

content +=  "\n\n"

  

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)

texts = text_splitter.split_text(content) # Splitting the document content into chunks

  
  

def  get_embedding(text, model="text-embedding-ada-002"): # Defining the function that creates the embeddings needed for the Chatbot to function (It can't form answers from plain text)

text = text.replace("\n", " ")

return openai.Embedding.create(input  = [text], model=model)['data'][0]['embedding']

  

df = pandas.read_csv(f"embs{g1}.csv") # Reading the empty csv file that you created earlier for storing the embeddings

  

df["combined"] = texts # Filling the 'combined' collumn with the chunks you created earlier

for i4 in  range(len(df["combined"])):

df["combined"][i4] =  '""'  + df["combined"][i4].replace("\n", "") +  '""'  # Adding triple quotes around the text chunks to prevent syntax errors caused by double quotes in the text

df.to_csv(f"embs{g1}.csv") # Writing the data to the csv file

  

df["embedding"] = df.combined.apply(lambda  x: get_embedding(x)) # Adding and filling the 'embedding' collumn which contains the embeddings created from your text chunks

df.to_csv(f"embs{g1}.csv", index=False) # Writing the new 'embedding' collumn to the csv file

  

df = pandas.read_csv(f"embs{g1}.csv") # Reading the new csv file

  

embs = []

for r1 in  range(len(df.embedding)): # Making the embeddings readable to the chatbot by turning it into a list

e1 = df.embedding[r1].split(",")

for ei2 in  range(len(e1)):

e1[ei2] =  float(e1[ei2].strip().replace("[", "").replace("]", ""))

embs.append(e1)

  

df["embedding"] = embs # Updating the 'embedding' collumn

df.to_csv(f"embs{g1}.csv", index=False) # Writing the final version of the csv file
```

For this tutorial, I'm using the "gpt-3.5-turbo" OpenAI model since it is the fastest and is the most cost efficient.

When you run your code, you would have prepared your data to be used by the chatbot, which means you can make the actual chatbot

