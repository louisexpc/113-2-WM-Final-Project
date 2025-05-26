from utils.import
def inference(system_content: str, user_content: str):

  messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
  ]
  try:
    input_ids = TOKENIZER.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(MODEL.device)
  except Exception as e:
      print(f"Error applying chat template: {e}")
      exit()

  terminators = [
      TOKENIZER.eos_token_id,
      TOKENIZER.convert_tokens_to_ids("<|eot_id|>")
  ]
  try:
    outputs = MODEL.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
  except Exception as e:
      print(f"Error during model generation: {e}")
      exit()
  response_ids = outputs[0][input_ids.shape[-1]:]
  response_text = TOKENIZER.decode(response_ids, skip_special_tokens=True)

  # print("\nAI Assistant:")
  # print(response_text)
  return response_text
