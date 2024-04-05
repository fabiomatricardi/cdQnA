'''
OPENVINO LLAMA2-7B MODEL REPOSITORY
https://huggingface.co/Advantech-EIOT/intel_llama-2-chat-7b/tree/main

CPU inference dOCS on HuggingFace
https://huggingface.co/docs/transformers/perf_infer_cpu?__cpo=aHR0cHM6Ly9odWdnaW5nZmFjZS5jbw


Installation of OPTIMUM
https://huggingface.co/docs/optimum/installation?__cpo=aHR0cHM6Ly9odWdnaW5nZmFjZS5jbw

➜ python -m venv venv
➜ .\venv\Scripts\activate
➜ pip install optimum
➜ pip install --upgrade-strategy eager optimum[openvino,nncf]
➜ pip install langchain
➜ pip install --upgrade pymupdf


Optimum Inference with OpenVINO
https://huggingface.co/docs/optimum/intel/inference?__cpo=aHR0cHM6Ly9odWdnaW5nZmFjZS5jbw

Optimum Intel can be used to load optimized models from the Hugging Face Hub and create pipelines to run inference with OpenVINO Runtime without rewriting your APIs.

CHAT TEMPLATING
https://huggingface.co/docs/transformers/main/chat_templating

https://huggingface.co/docs/transformers/main/chat_templating#is-there-an-automated-pipeline-for-chat

powershell -c "Invoke-WebRequest -Uri 'https://github.com/fabiomatricardi/Abstractive-Extractive/raw/main/images/old/stl-0000011.pdf' -OutFile 'stl-0000011.pdf'"


powershell -c "Invoke-WebRequest -Uri 'https://www.ias.edu/sites/default/files/library/UsefulnessHarpers.pdf' -OutFile 'UsefulnessHarpers.pdf'"

'''



import transformers
from optimum.intel.openvino import OVModelForCausalLM
import datetime
import tiktoken
encoding = tiktoken.get_encoding("r50k_base")
from tqdm.rich import trange, tqdm
from rich.markdown import Markdown
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from rich.console import Console
console = Console(width=110)


console.print('[red1 bold]Loading PDF...')
import os
#from keybert import KeyBERT
import fitz #pyMuPDF
miofile = "stl-0000011.pdf"
def LoadPDFandWork(filepath,chunks, overlap):
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
  return data,solotesto

d,article =  LoadPDFandWork(miofile, 2000,50)
context_count = len(encoding.encode(article))
console.print(f"Number of Tokens in the Article: {context_count}")
console.print('[red1 bold]---')
console.print('✅ Done')
console.print('[red bold]0.loading model...')
start = datetime.datetime.now()
tokenizer = transformers.AutoTokenizer.from_pretrained("Advantech-EIOT/intel_llama-2-chat-7b")
model = OVModelForCausalLM.from_pretrained("Advantech-EIOT/intel_llama-2-chat-7b")
delta = datetime.datetime.now() - start
print(f'model loaded in {delta}')
console.print('✅ Done')

"""
def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids,
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text
"""

console.print('[red bold]1.Generating Pipeline...')
pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)

context = article[:1200]
question = "what is the short summary of the text?"

def qna(pipeline, context, question):
    prompt = f'''Read this and answer the question.
[context]
{context}
[end of context]

Base your answer only on the given context.
If the question is unanswerable with the given context, ""say \"unanswerable\".
Question: {question}'''

    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always follows the instructions.",
        },
        {"role": "user", "content": prompt},
    ]
    #console.print('[cyan bold]4.Inference...')
    #console.print('[cyan bold]---')
    start = datetime.datetime.now()
    results = pipeline(messages, pad_token_id=pipeline.tokenizer.eos_token_id,max_new_tokens=350)[0]['generated_text'][-1]['content']
    delta = datetime.datetime.now() - start
    #console.print("[yellow]Response: ", results)
    #console.print('---')
    console.print(f'reply generated in {delta}')
    return results

generalQ = [
    "What is the title of the article?",
    "The article is written by? Who is the author?",
    "Who is Anne Frank?",
    "what is the web address of the article? http://"
]

replies = []
for item in generalQ:
  question = item
  answer = qna(pipe, context, item)
  if 'unanswerable' in answer.lower():
    console.print('😱no good')
  else:
    replies.append({
      'question' : question,
      'answer' : answer
  })

console.print('---')
console.print(replies)


from langchain.text_splitter import TokenTextSplitter
TOKENtext_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=50)
question_chunks = TOKENtext_splitter.split_text(article) #create a list
# context = Text to summarize:
def sumitall(pipeline,context,maxlenght):
    prompt = f"""Write a short summary of the given this text extracts:
[start of context]
{context}
[end of context]

Summary:"""
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always follows the instructions.",
        },
        {"role": "user", "content": prompt},
    ]
    #console.print('[cyan bold]4.Inference...')
    #console.print('[cyan bold]---')
    start = datetime.datetime.now()
    results = pipeline(messages, pad_token_id=pipeline.tokenizer.eos_token_id,max_new_tokens=maxlenght)[0]['generated_text'][-1]['content']
    delta = datetime.datetime.now() - start
    #console.print(f'SUMMARY>> {results}')
    console.print('---')
    console.print(f'Generated in {delta}')
    return results

partialSum = ''
for items in question_chunks:
  partialSum += sumitall(pipe,items,350) + '  '
console.print('---')
finalsummary = sumitall(pipe,partialSum,500)

replies.append({
    'question' : 'What is the summary?',
    'answer' : finalsummary
})
replies.append({
    'question' : 'What is the abstract of the article?',
    'answer' : finalsummary
})
replies.append({
    'question' : 'What is the article about?',
    'answer' : finalsummary
})
replies.append({
    'question' : 'What is this about?',
    'answer' : finalsummary
})
replies.append({
    'question' : 'What is this text about?',
    'answer' : finalsummary
})
