import json
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

CHUNK_FILE = "RAG/datasets/brain/ZZ/turn_data/turn_datasets.json"
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

model = SentenceTransformer("models/bge_large")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 去重，避免重复对导致 loss=0
seen = set()
train_examples = []
for item in dataset:
    for q in item["query"]:
        for p in item["positive"]:
            key = (q.strip(), p.strip())
            if key not in seen:
                seen.add(key)
                train_examples.append(InputExample(texts=[q, p]))

print(f"训练样本数: {len(train_examples)}")

# batch_size 加大，MNRL 依赖批内负例，越大越好
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=16,  # ← 32直接爆了
    collate_fn=model.smart_batching_collate
)

loss_fn = losses.MultipleNegativesRankingLoss(model)

# 学习率降低
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)  # ← 降一个数量级

NUM_EPOCHS = 3
total_steps = len(train_dataloader) * NUM_EPOCHS
warmup_steps = int(total_steps * 0.1)  # 前10%做warmup

# warmup + 线性衰减
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

loss_history = []
model.train()

for epoch in range(NUM_EPOCHS):
    for step, batch in enumerate(train_dataloader):
        sentence_features, labels = batch
        sentence_features = [
            {k: v.to(model.device) for k, v in sf.items()}
            for sf in sentence_features
        ]
        labels = labels.to(model.device)

        loss = loss_fn(sentence_features, labels)
        loss.backward()

        # 梯度裁剪，防止梯度爆炸导致震荡
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # ← 更新学习率
        optimizer.zero_grad()

        loss_history.append(loss.item())

        if step % 50 == 0:
            print(f"epoch {epoch}, step {step}, loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

# 平滑曲线（移动平均）
def smooth(data, weight=0.85):
    smoothed, last = [], data[0]
    for val in data:
        last = last * weight + val * (1 - weight)
        smoothed.append(last)
    return smoothed

plt.figure(figsize=(10, 4))
plt.plot(loss_history, alpha=0.3, label="raw")
plt.plot(smooth(loss_history), label="smoothed", linewidth=2)
plt.title("Training Loss Curve")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_curve.png")

model.save("models/bge_large_finetuned")