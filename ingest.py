"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle

# PATH_TO_DOCUMENTS = "Notion_DB/"
# INDEX_NAME = "notion.index"
# STORE_NAME = "notion.pkl"

PATH_TO_DOCUMENTS = "trainual/"
INDEX_NAME = "trainual.index"
STORE_NAME = "trainual.pkl"
# Here we load in the data in the format that Notion exports it in.
ps = list(Path(PATH_TO_DOCUMENTS).glob("**/*.md"))

data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(p)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, HuggingFaceEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, INDEX_NAME)
store.index = None
with open(STORE_NAME, "wb") as f:
    pickle.dump(store, f)


