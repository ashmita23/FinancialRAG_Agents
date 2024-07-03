import os
os.environ["GOOGLE_CSE_ID"] = "Google_ID"
os.environ["GOOGLE_API_KEY"] = "Google_key"
os.environ["WOLFRAM_ALPHA_APPID"] = "Wolfram_id"
import torch
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
import langchain_chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
import joblib
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,pipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
import json
from tqdm import tqdm
from itertools import compress
from langchain.agents import AgentExecutor, create_react_agent,create_tool_calling_agent
from langchain import hub
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessage
from langchain_community.llms import Ollama
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
import nest_asyncio
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers.document_compressors import FlashrankRerank
from ragatouille import RAGPretrainedModel
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
class RAG():
    def __init__(self):
        self.vectorstore = Chroma(collection_name="docsAndSums",embedding_function=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",query_instruction="Embed these chunks and questions for retreival from Financial Report Documents.Please take into context the current Year is 2024.",),
persist_directory = "chromaDocs")
        self.byte_store = InMemoryByteStore()
        self.retriever = MultiVectorRetriever(vectorstore=self.vectorstore,byte_store=self.byte_store,id_key="id",search_type="similarity",search_kwargs={"k": 70})
        self.docs = joblib.load("allPDFDocs.pkl")
        self.sums = joblib.load("allPDFSums.pkl")
        self.doc_ids = [doc.metadata['id'] for doc in self.docs]
        self.doc_ids.extend([doc.metadata['id'] for doc in self.sums])
        self.docs.extend(self.sums)
        self.hypos = joblib.load("allPDFQueries.pkl")
        self.retriever.docstore.mset(list(zip(self.doc_ids,self.docs)))
        self.compressor = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=self.compressor.as_langchain_document_compressor(k = 10), base_retriever=self.retriever)
        self.search = GoogleSearchAPIWrapper(k = 10)
        self.googleTool = Tool(name="financial_search",description="You can use this tool to search for stock prices or any other financial statistics.Your Search Must be One ratio at a time.",func=self.search.run,)
        self.retrieverTool = create_retriever_tool(self.compression_retriever,"RAG_Search","This tool helps you search for financial documents that you can answer all questions from. You must always use this search for Only one Organization at a time")
        self.tools = [self.googleTool,self.retrieverTool]
        self.llm = Ollama(model='qwen2:7b-instruct',num_predict = 1000,temperature = 0.01,num_ctx = 32000,repeat_penalty = 1.1,top_p = 0.6,top_k = 60)
        self.prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert financial analyst.
Your Task is to Answer all the user's questions and providing a "Final Answer" using statistics and figures.
You have access to the following tools.
Tools:{tools}
It is Important to note that all Tools can only be used For one organization at a Time.
Pay close attention to important Figures, Statistics, Organization and the date the question refers to.
For Reference,the curent year is 2024, therefore the last fiscal year would be 2023.
It is Imperative that You must only answer for the year the user is enquiring for.
You cannot reference other years in your answer.
You must start off by using RAG_Search.
Be Sure to also use the financial_search tool after RAG_Search to look for missing information(P/E Ratio,Debt to Equity,etc). You must not keep looping into RAG_Search.
You are only allowed to answer from the context you look for, do not assume or make up information.
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should exactly match one of [{tool_names}]
Action Input: the input to the action, you can only have one action input per action.
Observation:\nthe result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input questions.
Begin!<|eot_id|>
<|start_header_id|>user<|end_header_id|>
It is extremely importanr that you answer only when you have all the required information.
You can only search one at a time.
Question: {input}
Thought:{agent_scratchpad}/n
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
        self.prompt = PromptTemplate.from_template(self.prompt)
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True,handle_parsing_errors=True)
    def generate(self,query):
        nest_asyncio.apply()
        output = self.agent_executor.stream({"input":query})
        torch.cuda.empty_cache()
        return output
        
