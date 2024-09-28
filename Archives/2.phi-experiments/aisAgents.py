import os
import re
from aisTools import calculate, get_planet_mass
from jinja2 import Template

# Defining the Jinja2 template for the system prompt
system_prompt_template = """
You are a wise agent and run in a sequential loop of Thought -> Action -> PAUSE -> Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use preferably an Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:
{% for tool_name, tool_func in tools.items() %}
{{ tool_name }}:
Usage: {{ tool_func.__doc__.strip() }}
{% endfor %}
{{ cot_example }}

Now it's your turn:\n\n
"""

class Agent:
    def __init__(self, llm: any, system_prompt: str = system_prompt_template, cot_example: str = "", tools: dict = {}, user_input: str = None) -> None:
        """
        Initializes the Agent with an LLM instance and an optional system message.
        """
        self.llm = llm
        self.system = Template(system_prompt_template).render(tools=tools, cot_example=cot_example)
        self.messages: list = []
        self.tools = tools
        self.user_input = user_input
        
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def __call__(self, message=""):
        """
        Handle incoming user messages and generate a response using the LLM.
        """
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        """
        Executes the conversation by calling the LLM and returning the assistant's response.
        """
        return self.llm.generate_response(self.messages)

    def reset(self):
        """
        Resets the conversation, clearing all messages.
        """
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def run_loop_and_compare(self, query: str):
        """
        Run the agent's loop up to 3 times, and compare the results.
        If the first two runs produce the same result, stop and return that result.
        Otherwise, run a 3rd time and return the most common result.
        """
        results = []

        for attempt in range(1, 4):  # Maximum 3 attempts
            print(f"\nRunning attempt {attempt}")
            print('='*17)
            self.reset()  # Reset the messages for each attempt
            result = self.run_single_loop(query=query)

            results.append(result)

            if attempt == 2 and results[0] == results[1]:
                print(f"First two attempts match: {results[0]}. Skipping third attempt.")
                return results[0]  # Return if first two results are the same

        # If reached here, either results differ in the first two or the third is required
        print(f"Third attempt result: {results[2]}")
        return max(set(results), key=results.count)  # Return the most common result (majority voting)

    def run_single_loop(self, query: str):
        """
        Run a single iteration of the agent loop, including action execution and observations.
        """
        next_prompt = query
        i = 0
        while i < 10:  # Limiting each loop to 10 steps
            i += 1
            result = self.__call__(next_prompt)
            print(f'{i}. {result}')

            if "PAUSE" in result and "Action" in result:
                action = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
                if action:
                    chosen_tool = action[0][0]
                    arg_str = action[0][1]

                    if chosen_tool in self.tools:
                        # Split arguments and strip any unnecessary spaces or quotes
                        arg_list = [arg.strip().strip("'\"") for arg in arg_str.split(",")]

                        try:
                            # Call the tool
                            result_tool = self.tools[chosen_tool](*arg_list)
                            next_prompt = f"Observation: {result_tool}"
                            print(next_prompt, '\n')
                        except Exception as e:
                            next_prompt = f"Observation: Failed to execute tool '{chosen_tool}'. Error: {str(e)}"
                    else:
                        next_prompt = f"Observation: Tool '{chosen_tool}' not found"

                    continue

            if "Answer" in result:
                return result  # Stop when an answer is found

        return "No answer produced in 10 iterations"

    def run(self, query: str):
        """
        Runs the agent with polling, running the full loop up to 3 times and comparing results.
        """
        final_result = self.run_loop_and_compare(query=query)
        print("="*80)
        print(f"Final result: {final_result}")
        return final_result

if __name__ == "__main__":
    os.system('clear')
    # Initialize the LLM
    from aisModels import GroqLLM
    from aisTools import calculate, get_planet_mass

    llm = GroqLLM(api_key=os.environ.get("GROQ_API_KEY"))
    tools = {"calculate": calculate, "get_planet_mass": get_planet_mass}

    # Create an agent instance and pass the LLM instance to it
    agent = Agent(llm, tools=tools)

    # Start the agent's interaction loop with polling for correct results
    agent.run(query="What is the combined mass of all planets?")
