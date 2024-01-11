import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.tools import Tool, DuckDuckGoSearchRun, ArxivQueryRun, WikipediaQueryRun 
from langchain.utilities import WikipediaAPIWrapper 
from langchain.agents import initialize_agent, load_tools
from langchain.agents import AgentType 
from langchain.chains import LLMChain
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
import openai
import os
import warnings
import openai
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import LLMMathChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from io import StringIO
from langchain.schema.document import Document

warnings.filterwarnings('ignore')
count2 = 0
st.title("The Big Brain Generator")
st.subheader("created by Team B(ased)")
st.markdown("### Ever wanted to quickly summarise and practise your notes?")

noteType = st.text_input("Input your note type(Math/Science): ")

st.write("You have currently chosen {}. Please make sure you have a valid input!".format(noteType))

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file != None:
    num = st.text_input("How many questions would you like? ")
    if num.isdigit()==True and num != 0:
        num = int(num)
        st.write("Note: If the AI responds with gibberish or cuts off, please restart the AI.")

        
        
        #loader = TextLoader(uploaded_file.read().decode(),)
        if True:
            #documents = loader.load() # Loads the documents
            text_splitter = CharacterTextSplitter(chunk_size=1450, chunk_overlap=0)
            documents = [Document(page_content=x) for x in text_splitter.split_text(uploaded_file.read().decode())]
            
            #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            #string_data = stringio.read()
            #text_splitter = CharacterTextSplitter(chunk_size=1450, chunk_overlap=0) #Splits into chunks
            #texts = text_splitter.split_documents(documents)
            #Embedding model
            embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key
            )
            chroma_text = Chroma.from_documents(
                documents, 
                embeddings
            )

            qa_text_llm = ChatOpenAI( #Science teacher
                openai_api_key=openai_api_key,
                model='gpt-3.5-turbo'
            )
            qa_text = RetrievalQA.from_chain_type(
                llm=qa_text_llm, 
                chain_type="stuff", 
                retriever=chroma_text.as_retriever()
            )

            mathTeacher = LLMMathChain.from_llm(  #change tmr
                llm=qa_text_llm
            )

            def ai(mathcher):
                count = 0
                for i in range(num):
                    print("")
                    if count < int(num/2) or mathcher:
                        query = "Create an MCQ question about a random topic within this document and NOT questions asking whether or not the document contains certain topics or equations. List the choices numerically. Do not show the answers."
                    else:
                        query = "Create an Open Ended question about a random topic within this document and NOT questions asking whether or not the document contains certain topics or equations. Do not show the answers."
                    qn = (qa_text.run(query).strip())
                    st.write(qn)
                    answers = st.text_input("Please enter your answer:",key=count)
                    if len(answers) >= 1:
                        if count >= int(num/2) and not(mathcher):
                            query = f"Here is my answer {answers} for the question {qn}. Evaluate my answer, and give me tips for improvement."
                            if not(mathcher):
                                qn = (qa_text.run(query).strip())
                            else:
                                query = f"What is the correct option for the question {qn}?"
                                qn = (mathTeacher.run(query).strip())
                            st.write(qn)
                        else:
                            query = f"Option {answers} is my answer to the question {qn}, if it is NOT one of the options provided or if the option incorrectly answers the question show the correct option."
                            if not(mathcher):
                                qn = (qa_text.run(query).strip())
                                st.write(qn)
                            else:
                                try:
                                    qn = (mathTeacher.run(query).strip())
                                    st.write(qn)
                                except:
                                    qn = (qa_text.run(query).strip())
                                    st.write(qn)
                        #clearMemory()
                    count += 1

            if noteType.lower() == "math":
                ai(True)

            else:
                ai(False)
