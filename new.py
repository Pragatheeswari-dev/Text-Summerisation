import os
import streamlit as st
import pickle
import time
import langchain
import langchain_community
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
import datetime
from datetime import datetime
import dateutil.parser as dparser
import requests
from bs4 import BeautifulSoup
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
# from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
# load_dotenv()  # take environment variables from .env (especially open API key)
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

#from dotenv import load_dotenv,find_dotenv
#load_dotenv(find_dotenv())
#OpenAI_key = os.environ.get("OPEN_AI_KEY")
#print(OpenAI_key)
st.sidebar.title("About:")
st.sidebar.write(
        """
        This demo tool is designed to answer the questions efficiently and save time reading lengthy articles.  
         - It is implemented using the **LangChain framework**.  
         
         - Data from the URL is loaded and split into desired chunks, and then embeddings are generated and stored in Faiss, a vector database.
         
         - Faiss, provides both storage and optimised methods for performing similarity searches on high-dimensional vectors.
         
         - To generate answer, **gpt-3.5-turbo-instruct** LLM model from openAI is used.
         
         - Articles are also *tagged for sentiments, aggressiveness, language, and style* using openai.
         
         - The Title, Author Name, and Date were retrieved from the URL using the **'BeautifulSoup'**, a package commonly used for web scrapping.
       
        ### Further optimisation:
        
         - **Multiple Pre-trained LLMs** for specific tasks are published every week with their benchmarking results. with enough vector database storage and GPU resources, these LLMs can be optimised for several use cases.
                 
            - **Community Well-being**: Summarizing public health articles to provide timely updates and recommendations for community health and safety.
            
            - **Policy Development**: Analyzing and summarizing policy documents to identify key issues and trends, facilitating evidence-based decision-making.
            
            - **Governance**: Summarizing news articles and reports related to government activities and public affairs to monitor public sentiment and inform policy responses.

         
        
        """
    )

st.title("News Research Tool 📰")
st.write("  **Uncover Insights, Save Time ⏳**")
# st.sidebar.title("News Article URLs")
# st.subheader()
st.subheader("Try It Out:")
# urs = []
# for i in range(3):
# url1 = st.text_input(f"URL1",value="\n".join("https://www.governmentnews.com.au/queensland-says-no-to-new-olympic-stadium/"))
# url2 = st.text_input(f"URL2",value="\n".join("https://www.news.com.au/entertainment/celebrity-life/celebrity-deaths/ive-sadly-diedbestselling-author-announces-own-death/news-story/f7b145a9070d832b43946a126cfc3e4e"))
# url3 = st.text_input(f"URL3",value="\n".join("https://www.news.com.au/world/coronavirus/health/wear-a-mask-nsw-health-responds-to-a-rise-in-cases-in-light-of-new-subvariant-strains/news-story/90cad04f2a329d8730871c00b3dd00cc"))



# url1 = "https://www.governmentnews.com.au/queensland-says-no-to-new-olympic-stadium/"
# url2 = "https://www.news.com.au/entertainment/celebrity-life/celebrity-deaths/ive-sadly-diedbestselling-author-announces-own-death/news-story/f7b145a9070d832b43946a126cfc3e4e"
# url3 ="https://www.news.com.au/world/coronavirus/health/wear-a-mask-nsw-health-responds-to-a-rise-in-cases-in-light-of-new-subvariant-strains/news-story/90cad04f2a329d8730871c00b3dd00cc"
# urs.append(url1)
# urs.append(url2)
# urs.append(url3)

# urls = []
# for i in range(3):
#     url = st.text_input(f"URL {i}",value=urs[i])
#     urls.append(url)

# print("urls  ===",urls[0])


default_urls = [  
    "https://www.news.com.au/entertainment/celebrity-life/celebrity-deaths/ive-sadly-diedbestselling-author-announces-own-death/news-story/f7b145a9070d832b43946a126cfc3e4e",
    "https://www.governmentnews.com.au/report-finds-failings-all-round-in-victorias-commonwealth-games-debacle/",
   #  "https://www.pm.gov.au/media/parents-and-economy-benefit-latest-reform",
     "https://www.governmentnews.com.au/queensland-says-no-to-new-olympic-stadium/",
    "https://www.news.com.au/world/coronavirus/health/wear-a-mask-nsw-health-responds-to-a-rise-in-cases-in-light-of-new-subvariant-strains/news-story/90cad04f2a329d8730871c00b3dd00cc"
]

# # url_input = st.text_area("URL(s)", height=200, help="Enter one or more news article URLs separated by line breaks.")
# st.subheader("Try It Out:")
st.write("Paste the URL(s) of news articles you want to analyse below, or use the sample URLs provided:")

url_input = st.text_area("URL(s)", value="\n".join(default_urls), height=210, help="Enter one or more news article URLs separated by line breaks. You can also use the sample URLs provided below.")
urls = []
for url in url_input.split("\n"):
        if url.strip() != "":
            urls.append(url)
print("url: ", urls)

process_url_clicked = st.button("Process URLs")
#file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
# llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613", max_tokens=500)
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", max_tokens=500)
llm = OpenAI(temperature=0, max_tokens=500)

def text_preprocessing(u):
   #main_placeholder.text("Data Loading...Started...✅✅✅")
   loaders = UnstructuredURLLoader(u)
   data = loaders.load()
   print(len(data))
   text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200
      )
   docs = text_splitter.split_documents(data)
   print(len(docs))
   final_texts = ""
   for i in range(0,len(docs)):
    
    final_texts = final_texts + docs[i].page_content
    
   return final_texts

if process_url_clicked:
    # load data
    # print("url:   ",urls)
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    vectorstore_openai.save_local("faiss_store")
    main_placeholder.text("Enter the question...✅✅✅")

    # Save the FAISS index to a pickle file
    #with open(file_path, "wb") as f:
    #    pickle.dump(vectorstore_openai, f)

class Tags(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
                ...,
                description="describes how aggressive the statement is, the higher the number the more aggressive",
                enum=[1, 2, 3, 4, 5],
            )
    language: str = Field(
                ..., enum=["spanish", "english", "french", "german", "italian"]
            )
            # political_tendency: str
    style: str = Field(..., enum = ["formal","informal"])


# query = main_placeholder.text_input("Question: ")
default_q = ["Whats  the Gov news about Olymbics?",
        "Why did the author die?",
             "What does the report say about Commonwealth games?",
             "coronavirus news?"]
# col1 = st.columns(1)    
# with col1:
with st.popover("Sample questions to try"):
        sample_q = st.text_area("Sample questions to try",value="\n".join(default_q), height=140, help="Enter any questions or copy&pasate one from below.")

# with st.popover("Sample questions to try"):
#     sample_q = st.text_area("Sample questions to try",value="\n".join(default_q), height=100, help="Enter any questions or copy&pasate one from below.")


query = st.text_input("Question: ")
generate_answer_clicked = st.button("Generate Answer")

if query:
    vectorstore_openai = FAISS.load_local("faiss_store", OpenAIEmbeddings(),allow_dangerous_deserialization = True)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vectorstore_openai.as_retriever())
    #chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = vectorstore_openai.as_retriever())
    
    result = chain({"question": query}, return_only_outputs=True)
    # result will be a dictionary of this format --> {"answer": "", "sources": [] }
    st.header("Answer")
    st.write(result["answer"])

    # Display sources, if available
    sources = result.get("sources", "")
    print("sources---",result)
    print("sources---",result["sources"])
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)
            soup = BeautifulSoup(requests.get(source).content, 'html.parser')
            t = soup.find('title')
            title = t.get_text()
            author1 = soup.find('meta', {'name': 'author'})
            print("author1 = ", author1)
            if author1 is not None:
                author = author1["content"]
            else:
                author = " "
            print("author = ", author)
            datest = soup.find('meta', {'property': 'article:published_time'})
            if datest is not None:
                da = datest["content"]
                date = dparser.parse(da,fuzzy=True)
                date = str(date.date())
            else:
                date = " " 
                print("da = ", date)
            st.write(f"Title: {title}")
            st.write(f"Author: {author}")
            st.write(f"Date: {date}")
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", max_tokens=500)
            chain = create_tagging_chain_pydantic(Tags, llm)
            r = text_preprocessing([source])
            # print("r  ....=",r)
            st.write(chain.run(r))
