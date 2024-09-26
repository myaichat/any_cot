# any_cot
Chain of Thought implementation using different API vendors

inspiration: https://medium.com/@odhitom09/llamaberry-unlocking-advanced-chain-of-thought-in-ai-reasoning-85ac71f0e839

### Groq test

```
cot_models:
  first_turn:
      name: "llama-3.1-70b-versatile"
      api: "groq"
      user_prompt: |        
        {user_prompt}.      
  followup:
      name: "llama-3.1-70b-versatile"
      api: "groq"
  synthesis:  
      name: "llama-3.1-70b-versatile"
      api: "groq"


cot_prompt:
  first_turn: |
  ...
  ```


  ### Execute:

  ```
  python bot.py config\groq\groq_test.yaml
  ```


  ### Hugging face test
  (generated image prompt)

  ```
  cot_models:
  first_turn:
      name: "mistralai/Mistral-Nemo-Instruct-2407"
      api: "hugging_face"
      user_prompt: |        
        {user_prompt}
        return a detailed description of updated  image prompt in <fused_image> tags.      
  followup:
      name: "mistralai/Mistral-Nemo-Instruct-2407"
      api: "hugging_face"
  synthesis:  
      name: "mistralai/Mistral-Nemo-Instruct-2407"
      api: "hugging_face"


cot_prompt:
  first_turn: |
  ...
  ```


  ### Execute:

  ```
  python bot.py config\hugging_face\test.yaml
  ```

### Mixed test
Groq + HF (Groq as synthesizer)

```
cot_models:
  first_turn:
      name: "mistralai/Mistral-Nemo-Instruct-2407"
      api: "hugging_face"
      user_prompt: |        
        {user_prompt}
             
  followup:
      name: "mistralai/Mistral-Nemo-Instruct-2407"
      api: "hugging_face"
  synthesis:  
      name: "llama-3.1-70b-versatile"
      api: "groq"


cot_prompt:
  first_turn: |
  ...
  ```


  ### Execute:

  ```
  python bot.py config\mixed\mixed_test.yaml
  ```