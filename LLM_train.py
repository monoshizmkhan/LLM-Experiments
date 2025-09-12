from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import evaluate

def tokenize_function(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print("Data loaded.")

print("Tokenizing data...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
encoded_dataset = dataset.map(tokenize_function, batched=True)
print("Data ready for training.")

print("Initializing model and training parameters...")
model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
)
print("Ready to train.")

print("Training...")
trainer.train()
print("Model trained.")

print("Saving model...")
save_directory = "./my_saved_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
print("Model saved.")

tokenizer = AutoTokenizer.from_pretrained("./my_saved_model")
model = AutoModelForCausalLM.from_pretrained("./my_saved_model")

print("Evaluating model...")
perplexity = evaluate.load("perplexity")

example_texts = [
    "Once upon a time, there was a brave knight.",
    "In a galaxy far, far away, a new adventure began.",
    "It was a dark and stormy night."
]

results = perplexity.compute(model_id="./my_saved_model", predictions=example_texts)
print("Results:")

print(results)


