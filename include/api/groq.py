import os, time
import asyncio
import groq
from groq import AsyncGroq, Groq

from pprint import pprint as pp

import include.config.init_config as init_config 

apc = init_config.apc


class AsyncClient(AsyncGroq):   
    def __init__(self, api_key):
        super().__init__(api_key=api_key)



def get_client (api):
    clients=apc.clients
    if api not in clients:
        client_api = AsyncClient
        api_key = os.getenv(f"{api.upper()}_API_KEY")
        assert api_key, f"API key for '{api.upper()}_API_KEY' not found"
        clients[api] =  client_api(api_key)

    return clients[api]


async def call_llm(cot_model, messages: list,
                   temperature: float = 0.7,
                   max_tokens: int = 8000) -> str:
    """Call the Groq API."""
    api=cot_model['api']
    model=cot_model['name']
    client=get_client(api)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


async def generate_turn(query: str, previous_turns: list = None) -> str:
    """Generate a single turn of reasoning, considering previous turns if available."""
    is_first_turn = previous_turns is None or len(previous_turns) == 0
    cot_model=None
    if is_first_turn:
        cot_model=apc.cot_models['first_turn'][0]

        initial_system_prompt = apc.cot_prompt['first_turn']
        messages = [{
            "role": "system",
            "content": initial_system_prompt
        }, {
            "role": "user",
            "content": query
        }]
    else:
        cot_model=apc.cot_models['followup'][0]
        followup_system_prompt = apc.cot_prompt['followup']
        previous_content = "\n\n".join(previous_turns)
        messages = [{
            "role": "system",
            "content": followup_system_prompt
        }, {
            "role":
            "user",
            "content":
            f"Original Query: {query}\n\nPrevious Turns:\n{previous_content}\n\nProvide the next turn of reasoning."
        }]
    assert cot_model, f"No cot_model found for {is_first_turn}" 
    return await call_llm(cot_model, messages)


async def synthesize_turns(query: str, turns: list) -> str:
    """Synthesize multiple turns of reasoning into a final answer."""
    turns_text = "\n\n".join(
        [f"Turn {i+1}:\n{turn}" for i, turn in enumerate(turns)])
    synthesis_prompt = apc.cot_prompt['synthesis']
    messages = [{
        "role": "system",
        "content": synthesis_prompt
    }, {
        "role":
        "user",
        "content":
        f"Original Query: {query}\n\nTurns of Reasoning:\n{turns_text}"
    }]
    cot_model=apc.cot_models['synthesis'][0]

    return await call_llm(cot_model,messages)
