# Using this repo

# Creating a vectorized version of the content

We build a vectorized data store to allow the model
to retrieve information to answer the question we are asking it.
To build the data store (e.g. `trainual.index` and `trainual.pkl`)
use the `ingest.py` script:

```
python ingest.py
```

This will create the files `trainual.index` and `trainual.pkl`.

# How to ask the mode a question

The `qa.ipynb` contains the an initial block that:
1. Loads the LLM.
2. Loads the vector store.

Run this code block first to load the model. If you are using Llama
you might need to first create an account in HuggingFace and ask for
authorization to use it. For details see [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). In summary, you will need to:
1. Get approval from Meta
2. Get approval from HF
3. Create a read token from here : https://huggingface.co/settings/tokens
4. `pip install transformers``
5. execute huggingface-cli login and provide read token
6. Execute your code. It should work fine.


The second code block will prompt the user to ask a question and
the model will output an answer.