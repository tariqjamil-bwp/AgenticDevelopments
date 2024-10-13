import streamlit as st
from aisAgents import Agent
from aisTools import calculate
from ddg import ddg_search, ddg_news
from aisModels import GroqLLM
import os

# Define the Chain of Thought (COT) example
cot_example = """
Example session:

Question: Who is current president of USA?
Thought: I need to find the name of President of USA.
Action: ddg_search: 'who is President of USA'
PAUSE

You will be called again with this:

Observation: Joe Biden

If you have the answer, output it as the Answer.

Answer: The president of the United States is Joe Biden
"""

# Define prompt
prompt = """You are a helpful assistant. Respond politely to greeting words.
 Run in a sequential loop of Thought -> Action -> PAUSE -> Observation.
At the end of the loop, you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use preferably an Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.
"""

# Define tools
tools = {"calculate": calculate, "ddg_search": ddg_search, "ddg_news": ddg_news}

# Initialize the LLM
llm = GroqLLM(api_key=os.environ.get("GROQ_API_KEY"))

# Create an agent instance
agent = Agent(llm=llm, cot_example=cot_example, tools=tools, system_prompt=prompt)

# Streamlit application
def main():
    st.title("Agnetic Chatbot")

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Input field for user message
    user_input = st.text_area("You:", key="user_input", height=100)

    # Handle user input
    if st.button("Send"):
        if user_input.strip().lower() == '/bye':
            st.session_state.history.append(("User", user_input))
            st.session_state.history.append(("Bot", "Goodbye!"))
            st.session_state.user_input = ""  # Reset input field
        else:
            # Append user input to chat history
            st.session_state.history.append(("User", user_input))
            
            # Run the agent to get a response
            with st.spinner("Processing..."):
                result = agent.run(query=user_input)
            
            # Append the bot's response to the chat history
            st.session_state.history.append(("Bot", result))

            # Clear the input field after sending
            st.session_state.user_input = ""

    # Display conversation history
    st.write("### Chat History:")
    for role, message in st.session_state.history:
        if role == "User":
            st.write(f"**You**: {message}")
        else:
            st.write(f"**Bot**: {message}")

if __name__ == "__main__":
    main()
