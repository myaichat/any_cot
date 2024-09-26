import os, sys, time
import asyncio
import yaml 
import click
from pprint import pprint as pp
from os.path import join
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

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

console = Console()
       
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




#source: https://github.com/Jaimboh/Llamaberry-Chain-of-Thought-Reasoning-in-AI
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

                console.print(parsed_user_prompt, style="bold yellow")
                console.print(Panel(parsed_user_prompt, title="User prompt", title_align="left", border_style="white", style="green"))
                



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
    #print(result)
    console.print(Panel(result, title="Output", title_align="left", border_style="blue", style="white"))

if __name__ == "__main__":
    main()


"""
Displays:

╭─ User prompt ────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Justify importance of number 42.                                                                                     │
│                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ # Chain of Thought Reasoning                                                                                         │
│                                                                                                                      │
│ ## Original Query                                                                                                    │
│ Justify importance of number 42                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
│ ## Turn 1                                                                                                            │
│ ## Reasoning                                                                                                         │
│ 1. **Identify the source and context of the number 42**                                                              │
│    **Explanation:** The number 42 is famously referenced in "The Hitchhiker's Guide to the Galaxy" by Douglas Adams. │
│ In the story, the supercomputer Deep Thought is asked to find the answer to the Ultimate Question of Life, The       │
│ Universe, and Everything, to which it famously replies "42," though it admits that the question itself is unknown.   │
│                                                                                                                      │
│ 2. **Explore the significance of 42 within the narrative**                                                           │
│    **Explanation:** In the context of the story, the number 42 doesn't hold any inherent meaning or significance.    │
│ It's simply an arbitrary answer given by a supercomputer to a question that doesn't make sense. This is a commentary │
│ on the absurdity of trying to find meaning or purpose in life without knowing what the true purpose or question is.  │
│                                                                                                                      │
│ 3. **Investigate the cultural impact of the number 42**                                                              │
│    **Explanation:** Despite its arbitrary nature in the story, the number 42 has become culturally significant. It's │
│ often used as a reference point in popular culture, appearing in various movies, TV shows, and even in the name of a │
│ professional baseball team, the St. Louis Browns (who later became the Baltimore Orioles), which was a tribute to    │
│ the story.                                                                                                           │
│                                                                                                                      │
│ 4. **Examine the mathematical and scientific significance of 42**                                                    │
│    **Explanation:** While not particularly significant in most branches of mathematics or science, the number 42     │
│ does appear in some interesting contexts. For instance, it's the sum of the first six positive even numbers          │
│ (2+4+6+8+10+12), and it's also the largest number that can be expressed as the sum of two positive cubes (14^3 + 9^3 │
│ = 27466 + 729 = 42).                                                                                                 │
│                                                                                                                      │
│ ## Answer                                                                                                            │
│ The importance of the number 42 lies primarily in its cultural significance as a reference to "The Hitchhiker's      │
│ Guide to the Galaxy." Its arbitrary nature within the story serves as a commentary on the absurdity of seeking       │
│ meaning without understanding the question. While it does appear in some interesting mathematical and scientific     │
│ contexts, its most notable significance is in popular culture.                                                       │
│                                                                                                                      │
│ ## Turn 2                                                                                                            │
│ ## Critique                                                                                                          │
│ The previous reasoning effectively establishes the cultural significance of the number 42, anchored in its arbitrary │
│ yet iconic role in "The Hitchhiker's Guide to the Galaxy." The exploration of its mathematical and scientific        │
│ appearances is also notable, but it could be expanded to consider more aspects of these fields. The weakness in the  │
│ previous reasoning lies in not delving deeper into the mathematical and scientific significance, which might provide │
│ additional insights into the number's importance beyond its cultural impact.                                         │
│                                                                                                                      │
│ ## New Reasoning                                                                                                     │
│ 1. **Investigate the number 42 in the context of the Fibonacci sequence**                                            │
│    **Explanation:** The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding  │
│ ones, usually starting with 0 and 1. If we consider the sequence modulo 42, we find a pattern that repeats every 42  │
│ numbers. This is because 42 is a pseudoperfect number (a number equal to the sum of its proper divisors), and the    │
│ Fibonacci sequence modulo 42 has been extensively studied. This connection shows that 42 has a deeper mathematical   │
│ significance than previously mentioned.                                                                              │
│                                                                                                                      │
│ 2. **Explore the number 42 in the field of combinatorics**                                                           │
│    **Explanation:** In combinatorics, the number 42 appears in the formula for the number of derangements            │
│ (permutations where no element appears in its original position) of 9 elements, which is given by the subfactorial   │
│ function !9 = 41,830,880. This is a less well-known appearance of 42, but it demonstrates the number's significance  │
│ in another branch of mathematics.                                                                                    │
│                                                                                                                      │
│ 3. **Consider the number 42 in relation to the Golden Ratio**                                                        │
│    **Explanation:** The Golden Ratio, often denoted by the Greek letter Phi (Φ), is approximately equal to           │
│ 1.61803... If we consider the ratio of consecutive Fibonacci numbers, as the numbers get larger, this ratio          │
│ approaches the Golden Ratio. Interestingly, if we look at the 42nd and 43rd Fibonacci numbers, their ratio is very   │
│ close to the Golden Ratio: 1,023,341,55 (42nd Fibonacci number) / 632,459 (43rd Fibonacci number) ≈ 1.61803. This    │
│ further emphasizes the mathematical significance of the number 42.                                                   │
│                                                                                                                      │
│ ## Updated Answer                                                                                                    │
│ The importance of the number 42 is multifaceted, with its most recognized significance being its cultural impact as  │
│ a reference to "The Hitchhiker's Guide to the Galaxy." However, its mathematical and scientific significance is more │
│ extensive than previously discussed. The number 42 appears in the context of the Fibonacci sequence, combinatorics,  │
│ and its relationship with the Golden Ratio, indicating that it holds more profound mathematical relevance beyond its │
│ arbitrary role in the story. These additional aspects contribute to the broader importance and fascination with the  │
│ number 42.                                                                                                           │
│                                                                                                                      │
│ ## Turn 3                                                                                                            │
│ ## Next Reasoning                                                                                                    │
│                                                                                                                      │
│ 5. **Analyze the number 42 in the context of computer science and programming**                                      │
│    **Explanation:** In computer science, the number 42 has been adopted as an informal reference to the answer to    │
│ the Ultimate Question of Life, The Universe, and Everything. This has led to several interesting appearances of the  │
│ number in programming and computer science.                                                                          │
│                                                                                                                      │
│    - In the programming language Perl, the built-in constant for the number of arguments passed to a script is set   │
│ to 42 by default, reflecting this connection.                                                                        │
│                                                                                                                      │
│    - The computer science competition Codeforces uses 42 as its official problem ID for the first problem in its     │
│ training section, often referred to as "Problem 42."                                                                 │
│                                                                                                                      │
│    - In the field of cryptography, the number 42 is one of the fundamental blocks used in the RSA encryption         │
│ algorithm, which is based on the mathematical properties of prime numbers. This further emphasizes the number's      │
│ significance in computer science and mathematics.                                                                    │
│                                                                                                                      │
│ This additional reasoning highlights the number 42's relevance in the field of computer science and programming,     │
│ further expanding its importance beyond cultural, mathematical, and scientific contexts.                             │
│                                                                                                                      │
│ ## Synthesis                                                                                                         │
│ ## Analysis of Turns                                                                                                 │
│ Turn 1 primarily focuses on the cultural significance of the number 42, its origin in "The Hitchhiker's Guide to the │
│ Galaxy," and its appearances in mathematics and science. It provides a solid foundation for understanding the        │
│ number's importance but does not delve deeply into its mathematical and scientific aspects.                          │
│                                                                                                                      │
│ Turn 2 expands on the mathematical and scientific significance of the number 42, discussing its connection to the    │
│ Fibonacci sequence, combinatorics, and the Golden Ratio. This turn strengthens the argument for the number's         │
│ importance by exploring its role in various branches of mathematics.                                                 │
│                                                                                                                      │
│ Turn 3 introduces the number 42's relevance in computer science and programming, citing examples from Perl,          │
│ Codeforces, and cryptography. This turn highlights the number's impact on the field of computer science, building on │
│ the previous turns' discussions of mathematics and culture.                                                          │
│                                                                                                                      │
│ ## Comparison                                                                                                        │
│ All three turns share a common goal: to justify the importance of the number 42. However, they differ in their       │
│ approaches and the aspects of the number they emphasize. Turn 1 focuses on cultural significance, Turn 2 on          │
│ mathematical and scientific relevance, and Turn 3 on computer science and programming.                               │
│                                                                                                                      │
│ A notable similarity between Turns 2 and 3 is their exploration of the number 42's role in various fields beyond its │
│ cultural significance. These turns demonstrate that the number's importance extends beyond its iconic status in      │
│ popular culture.                                                                                                     │
│                                                                                                                      │
│ ## Final Reasoning                                                                                                   │
│ Combining the insights from all three turns, it becomes clear that the importance of the number 42 lies in its       │
│ multifaceted nature. The number holds cultural significance as a reference to "The Hitchhiker's Guide to the         │
│ Galaxy," mathematical and scientific relevance in various branches of mathematics, and significance in computer      │
│ science and programming.                                                                                             │
│                                                                                                                      │
│ The number 42's appearances in mathematics, such as its connection to the Fibonacci sequence, combinatorics, and the │
│ Golden Ratio, demonstrate its profound mathematical relevance. Its role in computer science and programming, as seen │
│ in Perl, Codeforces, and cryptography, further expands its importance.                                               │
│                                                                                                                      │
│ Moreover, the number 42's cultural impact is undeniable, with its iconic status in popular culture and its use as a  │
│ reference point in various movies, TV shows, and other media.                                                        │
│                                                                                                                      │
│ ## Comprehensive Final Answer                                                                                        │
│ The importance of the number 42 is multifaceted, encompassing its cultural significance as a reference to "The       │
│ Hitchhiker's Guide to the Galaxy," its mathematical and scientific relevance in various branches of mathematics, and │
│ its significance in computer science and programming. The number's appearances in mathematics, computer science, and │
│ popular culture demonstrate its profound impact across multiple fields, solidifying its status as a unique and       │
│ fascinating number.                                                                                                  │
│                                                                                                                      │
│ ## Concise Answer                                                                                                    │
│ The number 42 holds significance in multiple fields, including culture, mathematics, and computer science. Its       │
│ iconic status in popular culture, appearances in mathematics, and role in computer science and programming make it a │
│ fascinating and multifaceted number.                                                                                 │
│                                                                                                                      │
│ ## Timing Information                                                                                                │
│ - Turn 1 Time: 0.27s                                                                                                 │
│ - Turn 2 Time: 20.10s                                                                                                │
│ - Turn 3 Time: 6.39s                                                                                                 │
│ - Total Turns Time: 26.76s                                                                                           │
│ - Synthesis Time: 3.30s                                                                                              │
│ - Total Time: 30.06s                                                                                                 │
│                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
"""