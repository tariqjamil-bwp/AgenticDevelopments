import os
from datetime import date
from decouple import config

def set_environment_variables(project_name: str = "", env_path: str = None) -> None:
    if not project_name:
        project_name = f"Test_{date.today()}"

    cwd = os.getcwd()  # backup the path
    if env_path:
        os.chdir(envpath)
    else:
        abs_path = os.path.dirname(os.path.abspath(__file__))
        env_path = find_env_file(abs_path)
        os.chdir(env_path)
    
    print(f"reading environment variables from: <{os.getcwd()}>")
    os.environ["OPENAI_API_KEY"]       = str(config("OPENAI_API_KEY"))
    os.environ["LANGCHAIN_API_KEY"]    = str(config("LANGCHAIN_API_KEY"))
    os.environ["TAVILY_API_KEY"]       = str(config("TAVILY_API_KEY"))
    os.environ["WEATHER_API_KEY"]      = str(config("WEATHER_API_KEY"))
    os.environ["AGENTSOPS_API_KEY"]    = str(config("AGENTSOPS_API_KEY"))

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = project_name
    
    print("API Keys loaded and tracing set with project name: ", project_name)

    os.chdir(cwd)
########################################################################################################################
def find_env_file(start_path='.'):
    current_path = os.path.abspath(start_path)
    while True:
        env_path = os.path.join(current_path, '.env')
        if os.path.isfile(env_path):
            return os.path.dirname(env_path)

        parent_path = os.path.dirname(current_path)
        if current_path == parent_path:
            break
       
        current_path = parent_path

    print("Error: .env file not found")
    return None
########################################################################################################################
if __name__=="__main__":
    os.system("clear")
    set_environment_variables()