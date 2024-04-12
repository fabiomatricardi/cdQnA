#GENERAL CALL IMPORT
from tqdm.rich import trange, tqdm
from rich.markdown import Markdown
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from rich.console import Console
console = Console(width=110)
from llama_cpp import Llama
import tiktoken
encoding = tiktoken.get_encoding("r50k_base")



######## FUNCTIONS
#Load PDF Function
import os
import fitz #pyMuPDF
miofile = "/content/28884E00- SYSTEM OPERATIONAL TEST PROCEDURE PREPARATION CUIDELINE.pdf"
def LoadPDFandWork(filepath,chunks, overlap):
  """
  pass a file path, int chunk and overlap
  return a list d of text chunks and full article text
  """
  from langchain.document_loaders import TextLoader
  from langchain.text_splitter import TokenTextSplitter
  TOKENtext_splitter = TokenTextSplitter(chunk_size=chunks, chunk_overlap=overlap)
  #splitted_text = TOKENtext_splitter.split_text(fulltext) #create a list
  from langchain_community.document_loaders import PyMuPDFLoader
  import datetime
  start = datetime.datetime.now()
  console.print('1. loading pdf')
  loader = PyMuPDFLoader(filepath) #on Win local simply 'stl-0000011.pdf'
  data = loader.load_and_split(TOKENtext_splitter)
  delta = datetime.datetime.now() - start
  console.print(f'2. Loaded in {delta}')
  console.print(f'3. Number of items: {len(data)}')
  console.print('---')
  its = 0
  chars = 0
  solotesto = ''
  for items in data:
      testo = len(items.page_content)
      solotesto = solotesto + ' ' + items.page_content
      #console.print(f"Number of CHAR in Document {its}: {testo}")
      its = its + 1
      chars += testo

  console.print('---')
  console.print(f'> Total lenght of text in characthers: {chars}')
  console.print('---')
  context_count = len(encoding.encode(solotesto))
  console.print(f"Number of Tokens in the Article: {context_count}")
  d = []
  for items in data:
    d.append(items.page_content)
  return d,solotesto
"""
d,article =  LoadPDFandWork(miofile, 300,50)
"""

#lOAD A TXT FILE
#FOR TXT
# filename = '/content/2024-04-11 12.52.28 Kaggle s wrong turn when AI becomes a teacher and.txt'
def LoadTXT(filename, chunks, overlap):
    """
    pass a file path, int chunk and overlap
    return a list d of text chunks and full article text
    """
    with open(filename, encoding='utf-8') as f:
        article = f.read()
    f.close()
    import tiktoken
    encoding = tiktoken.get_encoding("r50k_base")
    context_count = len(encoding.encode(article))
    console.print(f"Number of Tokens in the Article: {context_count}")  
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import TokenTextSplitter
    TOKENtext_splitter = TokenTextSplitter(chunk_size=chunks, chunk_overlap=overlap)
    d = TOKENtext_splitter.split_text(article) #create a list
    console.print(f"Number of Document Chunks in the Article: {len(d)}") 
    return d, article
"""
d,article =  LoadTXT(miofile, 1200,50)
"""



