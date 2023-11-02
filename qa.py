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
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=local_llm, retriever=store.as_retriever(),
    return_source_documents=True
)
result = chain({"question": "How do I request vacation??"})
print(f"Answer: {result['answer']}")

# The output of the chain has three keys:
# - answer
# - sources
# - source_documents
#
# These are defined as the `output_keys` property of the `BaseQAWithSourcesChain`:
# https://github.com/langchain-ai/langchain/blob/8bd3ce59cd4ef80db5eb52922bb31588596939e1/libs/langchain/langchain/chains/qa_with_sources/base.py#L104

# Noticed that the sources are always empty but the source_documents are not.
# The following github issues talk about this:
# - https://github.com/langchain-ai/langchain/issues/5642
# - https://github.com/langchain-ai/langchain/issues/11387
# It is solved by this PR:
# - https://github.com/langchain-ai/langchain/pull/12556
print(f"Sources: {result['sources']}")