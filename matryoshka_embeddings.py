from sentence_transformers import SentenceTransformer, models, InputExample, datasets
from sentence_transformers.losses import CoSENTLoss, MatryoshkaLoss, MultipleNegativesRankingLoss
from datasets import load_dataset
import random 
import math


def build_embedding_model(model_name,embedding_dimension=768):
    word_embedding_model = models.Transformer(model_name, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_mode_cls_token=True,pooling_mode_mean_tokens=False)
    dense_layer=models.Dense(pooling_model.get_sentence_embedding_dimension(),embedding_dimension)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model,dense_layer])
    return model 

model_name="bert-base-uncased"
model=build_embedding_model(model_name)
# base_loss = CoSENTLoss(model=model)
# loss = MatryoshkaLoss(model=model, loss=base_loss, matryoshka_dims=[1024, 768, 512, 256, 128, 64])

dataset = load_dataset("embedding-data/QQP_triplets")
dataset = dataset["train"].train_test_split(test_size=0.3)

train_data=[]
for i in range(len(dataset["train"])):
    query=dataset["train"][i]["set"]["query"]
    pos=dataset["train"][i]["set"]["pos"][0]
    negative=random.choice(dataset["train"][i]["set"]["neg"])
    train_data.append(InputExample(texts=[query,pos,negative]))

train_batch_size=32
train_dataloader=datasets.NoDuplicatesDataLoader(train_data,batch_size=train_batch_size)
train_loss=MultipleNegativesRankingLoss(model)
train_loss=MatryoshkaLoss(model,train_loss,matryoshka_dims=[ 768, 512, 256, 128, 64])
num_epochs=3
warmup_steps= math.ceil(len(train_dataloader)*num_epochs*0.1)
model.fit(
    train_objectives=[(train_dataloader,train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    use_amp=False,
    output_path="output"
)

repo="repo_name"
model.push_to_hub(repo,token="<token>",commit_message="final_model_commit",exist_ok=True,replace_model_card=True,train_datasets=["embedding-data/QQP_triplets"])
