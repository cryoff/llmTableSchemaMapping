from langchain.embeddings import HuggingFaceEmbeddings

model_name = "ramybaly/ner_nerd_fine"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print(hf.embed_query("This is a test document."))
