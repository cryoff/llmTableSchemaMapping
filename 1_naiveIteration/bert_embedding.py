from transformers import BertTokenizer, BertModel
import torch


class BertEmbedding:

    def __init__(self) -> None:
        super().__init__()
        model_name = "bert-base-uncased"  # You can choose a different BERT model if needed
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_embedding(self, text: str):
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokens)
        embedding = outputs.last_hidden_state[0].mean(dim=0).numpy()
        print(embedding)


if __name__ == "__main__":
    bert_embedding: BertEmbedding = BertEmbedding()
    bert_embedding.get_embedding("This is a test sentence")
