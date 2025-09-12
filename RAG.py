from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
import evaluate

print("Loading pre-trained tokenizer and models...")

tokenizer = AutoTokenizer.from_pretrained("./my_saved_model")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("./my_saved_model")

VECTOR_DB = []
print("Reading text file to use for RAG...")
dataset = []
with open('Frankenstein.txt', 'r') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')


def tokenize_rag(example):
    tokens = tokenizer(example, truncation=True, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens["input_ids"]

def add_chunk_to_database(chunk):
  embedding = tokenize_rag(chunk)
  VECTOR_DB.append((chunk, embedding))

for chunk in dataset:
  add_chunk_to_database(chunk)

print("Frankenstein.txt loaded, chunked and tokenized.")

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
  query_embedding = tokenize_rag(query)
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  similarities.sort(key=lambda x: x[1], reverse=True)
  return similarities[:top_n]

question = "What did Victor Frankenstein do?"
print("Asking the pre-trained GPT2:", question)

retrieved_knowledge = retrieve(question, top_n=10)

instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''
print("Prompt:")
print(instruction_prompt)

print("#########")
print("Response:")
generator = pipeline("text-generation", model="gpt2")
output = generator(instruction_prompt)
print(output[0]["generated_text"])