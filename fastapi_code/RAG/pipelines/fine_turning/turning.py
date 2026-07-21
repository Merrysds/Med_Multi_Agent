from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json

CHUNK_FILE = "RAG/datasets/brain/ZZ/turn_data/turn_datasets.json"
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)
# for i, item in enumerate(dataset):
#     if "query" not in item:
#         print(i)
#         print(item)
#         break

model = SentenceTransformer("models/bge_large")

train_examples = []

for item in dataset:

    qs = item["query"]
    ps = item["positive"]

    for q in qs:
        for p in ps:
            train_examples.append(
                InputExample(texts=[q, p])
            )


train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=8
)

train_loss = losses.MultipleNegativesRankingLoss(model)

model.max_seq_length = 256

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    show_progress_bar=True,
    use_amp=True
)

model.save("models/bge_large_finetuned")