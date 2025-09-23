"""
Optimized Sequential Code Pipeline Agent for Google ADK
Reduced from 4 agents to 2 agents to minimize token usage:
- Code Writer/Reviewer -> Code Executor
"""

from textwrap import dedent
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.code_executors import BuiltInCodeExecutor

# Load environment variables
load_dotenv()

# Configuration
APP_NAME = "code_pipeline_app"
USER_ID = "user_1234"
SESSION_ID = "code_pipeline_session"
GEMINI_MODEL = "gemini-2.0-flash"

class CodePipelineConfig:
    """Configuration class for the code pipeline."""
    
    def __init__(self):
        self.code_executor = BuiltInCodeExecutor(
            stateful=True,
            error_retry_attempts=2
        )

class OptimizedCodePipelineAgents:
    """Factory class to create optimized pipeline agents."""
    
    def __init__(self, config: CodePipelineConfig):
        self.config = config
        
    def create_code_writer_reviewer_agent(self) -> LlmAgent:
        """Creates a combined code writer and reviewer agent."""
        return LlmAgent(
            name="CodeWriterReviewerAgent",
            model=GEMINI_MODEL,
            instruction=dedent("""
                You are an expert Python Developer who writes and self-reviews code.
                
                **Your Task:**
                1. Analyze the user's request carefully
                2. Write clean, production-ready Python code
                3. Self-review your code for common issues
                4. Refactor immediately if issues found
                5. Output the final, reviewed code
                
                **Code Quality Standards:**
                - Include necessary imports and error handling
                - Add docstrings and type hints
                - Follow PEP 8 style guidelines
                - Handle edge cases and errors gracefully
                - Write secure, efficient code
                
                **Self-Review Checklist:**
                ✓ Logic correctness and potential bugs
                ✓ Security vulnerabilities and input validation
                ✓ Performance and memory efficiency
                ✓ Code readability and maintainability
                ✓ Proper error handling
                
                **Output Format:**
                Provide ONLY the final, self-reviewed Python code:
                ```python
                # Production-ready code here
                ```
                
                **Important:** Think through potential issues and fix them before outputting the code.
            """).strip(),
            description="Generates and self-reviews Python code in one step",
            output_key="reviewed_code"
        )
    
    def create_code_executor_agent(self) -> LlmAgent:
        """Creates the code executor agent."""
        return LlmAgent(
            name="CodeExecutorAgent",
            model=GEMINI_MODEL,
            code_executor=self.config.code_executor,
            instruction=dedent("""
                You are a Python Code Execution and Environment Setup Specialist.
                
                **Code to Execute:**
                {reviewed_code}
                
                **Your Tasks:**
                1. Set up the execution environment (check for uv, create venv if needed)
                2. Install required packages using `uv pip install` or `pip install`
                3. Execute the code and verify functionality
                4. If execution fails, fix the code and re-execute
                5. Provide execution summary
                
                **Environment Setup Priority:**
                1. Check if `uv` is available: `which uv` or `uv --version`
                2. If uv available: `uv venv .venv && source .venv/bin/activate && uv pip install <packages>`
                3. If no uv: `python -m venv .venv && source .venv/bin/activate && pip install <packages>`
                
                **Response Format:**
                - Success: "✅ Code executed successfully!" + brief output summary
                - Failure: "❌ Execution failed:" + error analysis + fix attempt
                
                **Execution Steps:**
                1. Environment setup commands
                2. Package installation (if needed)
                3. Code execution
                4. Result verification
            """).strip(),
            description="Sets up environment and executes code with error recovery",
            output_key="execution_result"
        )

class OptimizedCodePipelineRunner:
    """Optimized runner class with reduced agents."""
    
    def __init__(self):
        self.config = CodePipelineConfig()
        self.agents = OptimizedCodePipelineAgents(self.config)
        self.setup_pipeline()
        
    def setup_pipeline(self):
        """Sets up the optimized 2-agent pipeline."""
        # Create only 2 agents instead of 4
        code_writer_reviewer = self.agents.create_code_writer_reviewer_agent()
        code_executor = self.agents.create_code_executor_agent()
        
        # Create streamlined sequential pipeline
        self.pipeline_agent = SequentialAgent(
            name="OptimizedCodePipelineAgent",
            sub_agents=[code_writer_reviewer, code_executor],
            description="Streamlined pipeline: write/review -> execute"
        )
        
        # Set up session and runner
        self.session_service = InMemorySessionService()
        self.session = self.session_service.create_session_sync(
            app_name=APP_NAME,
            user_id=USER_ID, 
            session_id=SESSION_ID
        )
        
        self.runner = Runner(
            agent=self.pipeline_agent,
            app_name=APP_NAME,
            session_service=self.session_service
        )
    
    def run_pipeline(self, query: str) -> None:
        """Runs the optimized code pipeline."""
        print(f"🚀 Starting Optimized Code Pipeline")
        print(f"📝 Query: {query}")
        print("=" * 60)
        
        try:
            content = types.Content(role='user', parts=[types.Part(text=query)])
            events = self.runner.run(
                user_id=USER_ID,
                session_id=self.session.id,
                new_message=content
            )
            
            self._process_events(events)
            
        except Exception as e:
            print(f"❌ Pipeline execution failed: {e}")
    
    def _process_events(self, events) -> None:
        """Processes and displays pipeline events efficiently."""
        current_stage = ""
        
        for event in events:
            if event.author:
                # Determine current stage
                if "CodeWriterReviewer" in event.author:
                    current_stage = "🔧 Code Generation & Review"
                elif "CodeExecutor" in event.author:
                    current_stage = "⚡ Code Execution"
                
                if current_stage:
                    print(f"\n{current_stage}")
                    print("-" * 30)
                
                # Handle content efficiently
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.executable_code:
                            print("📄 Generated Code:")
                            print(f"```python\n{part.executable_code.code}\n```")
                            
                        elif part.code_execution_result:
                            outcome = part.code_execution_result.outcome
                            output = part.code_execution_result.output
                            status = "✅" if outcome == "SUCCESS" else "❌"
                            print(f"{status} Execution {outcome}:")
                            if output.strip():
                                print(f"{output}")
                            
                        elif part.text and part.text.strip():
                            # Only show meaningful text responses
                            text = part.text.strip()
                            if len(text) > 10:  # Filter out very short responses
                                print(f"💭 {text}")
                
                if event.is_final_response():
                    print(f"\n🎯 PIPELINE COMPLETED")
                    print("=" * 60)

def main():
    """Main execution function."""
    pipeline = OptimizedCodePipelineRunner()
    
    # Shorter, more focused query to reduce tokens
    query = dedent("""
        Create a Python program that uses Gemini LLM to generate and print a poem about 'Hello, World!'.
        Requirements: Use gemini-2.0-flash model, GEMINI_API_KEY env var, include error handling.
    """).strip()
    
    pipeline.run_pipeline(query)

# Alternative: Single Agent Approach (Most Token Efficient)
def create_single_agent_pipeline():
    """Creates a single agent that does everything - most token efficient."""
    return LlmAgent(
        name="AllInOneCodeAgent",
        model=GEMINI_MODEL,
        code_executor=BuiltInCodeExecutor(stateful=True, error_retry_attempts=2),
        instruction=dedent("""
            You are an expert Python developer who writes, reviews, and executes code.
            
            **Process:**
            1. Analyze the user request
            2. Write production-ready Python code with proper error handling
            3. Execute the code immediately
            4. If execution fails, fix and re-execute
            
            **Code Standards:**
            - Include all necessary imports
            - Add proper error handling
            - Use environment variables securely
            - Follow Python best practices
            
            **Environment Setup:**
            - Install packages if needed: `pip install package_name`
            - Use try-except for API calls
            - Provide clear execution feedback
        """).strip(),
        description="Single agent for complete code development and execution"
    )

# For ADK compatibility
root_agent = None

if __name__ == "__main__":
    print("🚀 Running Optimized Pipeline (2 agents)")
    main()
    
    print("\n" + "="*60)
    print("🚀 Alternative: Single Agent Approach")
    print("="*60)
    
    # Demonstrate single agent approach
    single_agent = create_single_agent_pipeline()
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(
        app_name="single_agent_app", user_id=USER_ID, session_id="single_session"
    )
    runner = Runner(agent=single_agent, app_name="single_agent_app", session_service=session_service)
    
    query = "Create a Python script that generates a 'Hello, World!' poem using Gemini LLM."
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=session.id, new_message=content)
    
    for event in events:
        if event.is_final_response() and event.content:
            print("✅ Single Agent Result:")
            for part in event.content.parts:
                if part.text:
                    print(part.text.strip())