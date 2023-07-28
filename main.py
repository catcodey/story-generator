from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2")

def generate_short_story(initial_sentence, max_length=100, num_return_sequences=3):
    # Encode initial sentence
    input_ids = tokenizer.encode(initial_sentence, return_tensors="tf")

    # Generate scenarios
    scenarios = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.9,
    )

    # Decode generated scenarios
    generated_scenarios = [tokenizer.decode(scenario, skip_special_tokens=True) for scenario in scenarios]

    return generated_scenarios
def main():
    # User Input
    initial_sentence = input("Enter a small initial sentence to generate a short sweet story: ")

    # Generate scenarios
    generated_scenarios = generate_short_story(initial_sentence)

    # Output scenarios
    print("Generated Short Stories:\n")
    for i, scenario in enumerate(generated_scenarios):
        print(f"Story {i+1}\n: {scenario}")
        print("\n")
if __name__ == "__main__":
    main()
