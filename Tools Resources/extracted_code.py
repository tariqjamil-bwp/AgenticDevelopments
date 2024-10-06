if show_token_consumption:
print(colored("\nTokens used this time: " + str(tokens_used), 'red')
print(colored("Total tokens used so far: "+ str(total_session_token|
user_input = None
action = responses.split("Action: ")[1].split("\n")[@].strip()
action_input = responses.split("Action Input: ")[1].split("\n") [0].str]
if action == "Search":
tool = search
elif action == "Calculator":
tool = calculator
elif action == "Save response to csv":
tool = save_response_to_csv_file
elif action == "Response To Human":
print(f"Response: {action_input}")
user_input = input("enter your response: ")
total_session_tokens += Len(encoding.encode(user_input))
if tool:
observation = tool(action_input)
print("Observation: " , observation)
if user_input:
messages.extend([
{"role": "system" , "content": responses},
» “"content":user_input},])


def agent_Loop(initial_command) :
messages = [
{ "role": "system", "content": PROMPT },
{ "role": "user", "content": initial_command },
1
total_session_tokens = sum([len(encoding.encode(message["content"]))
for message in messages])
while True:
tool = None
response = openai.ChatCompletion.create(
model="gpt-4",
temperature=0.3,
stream=True,
messages=messages,
)
tokens_used = @
response = ''
#process each chunk
for chunk in response:
‘if "role" in chunk{"choices"] [@] ["delta"]:
continue
elif "content" in chunk["choices"] [0] ("delta")
tokens_used += 1
ratext = chuck["choic
responses += r_text
print(colored(r_text, 'green'), end=
total_session_tokens += tokens_used
if show_token_consumption:

"} [0] ["deLta"] ("content"}

*,flush=True)

def

def


