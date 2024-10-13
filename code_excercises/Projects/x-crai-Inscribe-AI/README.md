## Overview

This project demonstrates a content automation system that integrates advanced AI models to plan, write, and edit engaging and factually accurate blog posts. Using a combination of `crewai`, `LangChain`, and OpenAI’s LLM, the system automates content creation for blog articles, including generating content plans, writing posts, and editing them.

## Features

- **Content Planning:** Automatically generate a detailed content plan based on the provided topic, including audience analysis and SEO keywords.
- **Content Writing:** Craft compelling and structured blog posts using the provided content plan, ensuring natural incorporation of SEO keywords and engaging content.
- **Content Editing:** Proofread and edit blog posts to ensure alignment with the brand’s voice and correction of grammatical errors.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/content-automation-project.git
   cd content-automation-project
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   - Make sure to set the `OPENAI_API_KEY` environment variable with your OpenAI API key:
     ```bash
     export OPENAI_API_KEY="your_openai_api_key"  # On Windows, use `set` instead of `export`
     ```

## Usage

1. **Run the Content Automation Script:**
   ```bash
   python main.py
   ```

2. **Input the Topic:**
   - When prompted, enter the topic you want to generate content for.

## Example

```python
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Initialize LLM
llm = ChatOpenAI(model="crewai-llama3", base_url="http://localhost:11434/v1")

# Define Agents
planner = Agent(role="Content Planner", goal="Plan engaging content", llm=llm)
writer = Agent(role="Content Writer", goal="Write insightful content", llm=llm)
editor = Agent(role="Editor", goal="Edit blog post", llm=llm)

# Define Tasks
plan = Task(description="Plan content outline", agent=planner)
write = Task(description="Write blog post", agent=writer)
edit = Task(description="Edit blog post", agent=editor)

# Initialize Crew
crew = Crew(agents=[planner, writer, editor], tasks=[plan, write, edit])

# Execute
inputs = {"topic": "Comparative study of LangGraph, Autogen, and Crewai for building multi-agent systems."}
result = crew.kickoff(inputs=inputs)
print(result)
```

Use Pandoc to convert from markdown to word
## Contributing

Feel free to contribute by submitting issues or pull requests. Ensure your contributions adhere to the project's coding standards and include appropriate tests.

## Acknowledgments

- [CrewAI](https://crewai.com) for providing the tools for content automation.
- [LangChain](https://langchain.com) for enabling advanced language model integrations.
- [OpenAI](https://openai.com) for their powerful language models.

---

Feel free to adjust the content according to your specific project details and needs.
