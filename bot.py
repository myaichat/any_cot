import os, sys, time
import asyncio
import yaml 
import click
from pprint import pprint as pp
from os.path import join
#from  include.api.deepinfra import AsyncClient, get_final_stream
import include.config.init_config as init_config 

init_config.init(**{})
apc = init_config.apc
apc.models={}


#import include.api.deepinfra as deepinfra
import include.api.groq as groq
#import include.api.together as together
#import include.api.openai as openai 
#import include.api.mistral as mistral
#import include.api.nvidia as nvidia
#import include.api.deepseek as deepseek
import include.api.hugging_face as hugging_face
#import include.api.anthropic as anthropic
#import include.api.gemini as gemini
#import include.api.cohere as cohere
#import include.api.palm2 as palm2
from include.common import get_aggregator 
e=sys.exit

apc.clients={}


       
async def close_clients():

    for client in apc.clients.values():
        await client.close()

def save_models(cot_models):   
    apc.cot_models=cot_models 
    for type, model in cot_models.items():
        api=model['api']
        apc.apis[api]=globals()[api]

def save_prompt(cot_prompt):   
    apc.cot_prompt=cot_prompt 


def get_prompt(model_prompt, user_prompt):
    parsed_string = model_prompt.format(user_prompt=user_prompt)
   
    return parsed_string





async def generate_turn(query: str, previous_turns: list = None) -> str:
    """Generate a single turn of reasoning, considering previous turns if available."""
    is_first_turn = previous_turns is None or len(previous_turns) == 0
    cot_model=None
    if is_first_turn:
        cot_model=apc.cot_models['first_turn']
        api=cot_model['api']
        initial_system_prompt = apc.cot_prompt['first_turn']
        messages = [{
            "role": "system",
            "content": initial_system_prompt
        }, {
            "role": "user",
            "content": query
        }]
    else:
        cot_model=apc.cot_models['followup']
        api=cot_model['api']
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
    return await apc.apis[api].call_llm(cot_model, messages)


async def synthesize_turns( query: str, turns: list) -> str:
    """Synthesize multiple turns of reasoning into a final answer."""
    turns_text = "\n\n".join(
        [f"Turn {i+1}:\n{turn}" for i, turn in enumerate(turns)])
    synthesis_prompt = apc.cot_prompt['synthesis']
    api=apc.cot_models['synthesis']['api']
    messages = [{
        "role": "system",
        "content": synthesis_prompt
    }, {
        "role":
        "user",
        "content":
        f"Original Query: {query}\n\nTurns of Reasoning:\n{turns_text}"
    }]
    cot_model=apc.cot_models['synthesis']

    return await apc.apis[api].call_llm(cot_model,messages)


async def full_cot_reasoning(query: str) -> tuple:
    """Perform full Chain of Thought reasoning with multiple turns."""
    start_time = time.time()
    turns = []
    turn_times = []
    full_output = f"# Chain of Thought Reasoning\n\n## Original Query\n{query}\n\n"
    if 1:
        for i in range(3):  # Generate 3 turns of reasoning
            turn_start = time.time()
            turn = await generate_turn(query, turns)
            turns.append(turn)
            turn_times.append(time.time() - turn_start)
            full_output += f"## Turn {i+1}\n{turn}\n\n"

    mid_time = time.time()
    synthesis = await synthesize_turns(query, turns)
    
    full_output += f"## Synthesis\n{synthesis}\n\n"
    end_time = time.time()

    timing = {
        'turn_times': turn_times,
        'total_turns_time': mid_time - start_time,
        'synthesis_time': end_time - mid_time,
        'total_time': end_time - start_time
    }

    full_output += f"## Timing Information\n"
    full_output += f"- Turn 1 Time: {timing['turn_times'][0]:.2f}s\n"
    full_output += f"- Turn 2 Time: {timing['turn_times'][1]:.2f}s\n"
    full_output += f"- Turn 3 Time: {timing['turn_times'][2]:.2f}s\n"
    full_output += f"- Total Turns Time: {timing['total_turns_time']:.2f}s\n"
    full_output += f"- Synthesis Time: {timing['synthesis_time']:.2f}s\n"
    full_output += f"- Total Time: {timing['total_time']:.2f}s\n"

    return full_output



@click.command()
@click.argument('yaml_file_path', type=click.Path(exists=True))


def main(yaml_file_path):
    async def async_main():
        """Run the main loop of the MOA process."""
        with open(yaml_file_path, 'r') as file:
            apc.pipeline = data = yaml.safe_load(file)
            apc.prompt_log['pipeline']=apc.pipeline
            
        apc.prompt_log['cot_models']={}    
        #pp(data)
        if cot_models := data.get('cot_models', None  ):
            save_models(cot_models)
        else:
            raise Exception('No cot_models found')
        if cot_prompt := data.get('cot_prompt', None  ):
            save_prompt(cot_prompt)
        else:
            raise Exception('No cot_models found')            


        apc.prompt_log['pipeline']={'models':cot_models}
        print("Running main loop...")


        try:
            while True:

                default_prompt="Justify importance of number 42"
                user_prompt = input(f"Enter your prompt ({default_prompt}): ")
                if not user_prompt:
                    user_prompt = default_prompt


                apc.prompt_log['pipeline']['user_prompt']=user_prompt
                cot_model_prompt=apc.cot_models['first_turn'].get('user_prompt', None)
                if cot_model_prompt:
                      
                    parsed_user_prompt = get_prompt(cot_model_prompt, user_prompt)
                else:
                    parsed_user_prompt=user_prompt

                apc.prompt_log['pipeline']['parsed_user_prompt']=parsed_user_prompt
                results={}



                """Wrapper to run the full CoT reasoning and display results."""
                out = await full_cot_reasoning(parsed_user_prompt)


                print()
                apc.prompt_log['final_stream']={}
                apc.prompt_log['final_stream']['result']=out
                apc.prompt_log['result'] = ' '.join(out)
                return out
        finally:
            await close_clients()
    result= asyncio.run(async_main())
    print(result)
if __name__ == "__main__":
    main()
