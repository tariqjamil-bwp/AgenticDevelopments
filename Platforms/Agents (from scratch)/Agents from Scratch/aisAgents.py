import os
import re
from aisTools import calculate, get_planet_mass
from jinja2 import Template
#####################################################FIX################################################################################
# Defining the Jinja2 template for the system prompt
system_prompt_template = """
You are a wise agent and run in a sequential loop of Thought -> Action -> PAUSE -> Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use preferrably an Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:
{% for tool_name, tool_func in tools.items() %}
{{ tool_name }}:
Usage: {{ tool_func.__doc__.strip() }}
{% endfor %}
{{ cot_example }}

Now it's your turn:\n\n
"""
#####################################################FIX################################################################################
class Agent:
    def __init__(self, llm: any, system_prompt: str = system_prompt_template, cot_example: str = "", tools: dict = {}, user_input: str = None) -> None:
        """
        Initializes the Agent with an LLM instance and an optional system message.
        
        :param llm: An instance of the LLM class used to generate responses.
        :param system_prompt: An optional system message to initialize the conversation context.
        """
        self.llm = llm
        self.system = Template(system_prompt_template).render(tools=tools, cot_example=cot_example)
        self.messages: list = []
        self.tools = tools
        self.user_input = user_input
        #print(self.system)
        #print(90*"-")
        
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def __call__(self, message=""):
        """
        Handle incoming user messages and generate a response using the LLM.
        
        :param message: The user's message to be added to the conversation.
        :return: The assistant's response to the user.
        """
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        """
        Executes the conversation by calling the LLM and returning the assistant's response.
        
        :return: The generated response from the model.
        """
        return self.llm.generate_response(self.messages)

    def reset(self):
        """
        Resets the conversation, clearing all messages.
        """
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def run(self, max_iterations=10, query: str = "", tools=None, user_input=None):
        """
        Runs a loop to interact with the agent, process tools, and handle the conversation.
        
        :param max_iterations: The maximum number of iterations for the loop.
        :param query: The initial user query to start the loop.
        :param tools: A dictionary of available tool names mapped to their functions.
        """
        if tools is None:
            tools = self.tools

        if user_input is None:
            user_input = self.user_input

        next_prompt = query
        i = 0

        while i < max_iterations:
            i += 1
            result = self.__call__(next_prompt)
            print(f'{i}. {result}')

            if "PAUSE" in result and "Action" in result:
                action = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
                if action:
                    chosen_tool = action[0][0]
                    arg_str = action[0][1]

                    if chosen_tool in tools:
                        # Split arguments and strip any unnecessary spaces or quotes
                        arg_list = [arg.strip().strip("'\"") for arg in arg_str.split(",")]
                        
                        try:
                            print(f'Calling: {chosen_tool}({arg_str})', end='')
                            result_tool = tools[chosen_tool](*arg_list)
                            next_prompt = f"Observation: {result_tool}"
                            
                        except Exception as e:
                            next_prompt = f"Observation: Failed to execute tool '{chosen_tool}'. Error: {str(e)}"
                    else:
                        next_prompt = f"Observation: Tool '{chosen_tool}' not found"

                    print(next_prompt, '\n')
                    continue

            if "Answer" in result:
                break

            # Optional: Add a user-driven exit condition
            if self.user_input:
                user_input = input("Continue? (type 'exit' to stop): ").strip().lower()
                if user_input == 'exit':
                    break

if __name__ == "__main__":
    import os
    os.system('clear')
    # Initialize the LLM
    from aisModels import GroqLLM
    from aisTools import calculate, get_planet_mass

    llm = GroqLLM(api_key=os.environ.get("GROQ_API_KEY"))
    tools = {"calculate": calculate, "get_planet_mass": get_planet_mass}
    # Create an agent instance and pass the LLM instance to it
    agent = Agent(llm, tools=tools)

    # Start the agent's interaction loop
    agent.run(max_iterations=20, query="What is the combined mass of all planets?")