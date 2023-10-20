import faiss
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import HuggingFacePipeline

# To Download LLAMA: https://github.com/facebookresearch/llama/issues/374#issuecomment-1643228958
model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
#     max_length=3773,
#     temperature=0,
#     top_p=0.95,
#     repetition_penalty=1.15,
)

local_llm = HuggingFacePipeline(pipeline=pipe)

#print(local_llm('translate English to German: how old are you?'))

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=local_llm, retriever=store.as_retriever(), return_source_documents=True)
result = chain({"question": "How do I request vacation??"})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")