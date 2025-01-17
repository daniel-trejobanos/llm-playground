from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the Meditron model and tokenizer
model_name = "epfl-llm/meditron-7b"  # Replace with the desired Meditron model size
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_cot_response(prompt):
  """
  Generates a Chain-of-Thought (CoT) response using the Meditron model.

  Args:
    prompt: The input question or task.

  Returns:
    The generated CoT response.
  """

  # 1. Prompt the model for a CoT explanation:
  cot_prompt = f"Let's think step-by-step. {prompt}"
  input_ids = tokenizer(cot_prompt, return_tensors="pt").input_ids

  # 2. Generate the CoT response:
  generated_ids = model.generate(
      input_ids=input_ids,
      max_length=256,  # Adjust as needed
      num_beams=5,     # Adjust for beam search
      no_repeat_ngram_size=2,
      early_stopping=True
  )

  # 3. Decode the generated response:
  cot_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

  return cot_response

# Example usage:
question = "What is the capital of France?"
cot_response = generate_cot_response(question)
print(cot_response)
