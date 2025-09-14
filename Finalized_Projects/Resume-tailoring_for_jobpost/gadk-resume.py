# -*- coding: utf-8 -*-
"""
Google ADK Job Application Multi-Agent System
Converted from CrewAI to Google ADK for tailoring job applications
"""

import os
import asyncio
import json
import httpx
from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# Constants and API Key Checks
# -----------------------------
APP_NAME = "job_application_app"
USER_ID = "user_001"
SESSION_ID = "job_application_session_001"

if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY environment variable not set.")
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# -----------------------------
# LLM Model
# -----------------------------
AGENT_MODEL = LiteLlm("gemini/gemini-2.0-flash")

# -----------------------------
# Define Asynchronous Tool Functions
# -----------------------------
async def search_internet(query: str) -> str:
    """Search the internet for job-related information."""
    print(f"--- Using Tool: search_internet with query: '{query}' ---")
    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {'X-API-KEY': os.environ['SERPER_API_KEY'], 'Content-Type': 'application/json'}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, content=payload)
        response.raise_for_status()
        response_json = response.json()
        if 'organic' not in response_json or not response_json['organic']: 
            return "No organic results found."
        results = response_json['organic'][:4]
        string = [f"Title: {r.get('title', 'N/A')}\nLink: {r.get('link', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')}\n---" for r in results]
        return '\n'.join(string)
    except Exception as e:
        return f"An error occurred during search: {e}"

async def scrape_website(website_url: str) -> str:
    """Scrape content from a website URL."""
    print(f"--- Using Tool: scrape_website with URL: '{website_url}' ---")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(website_url)
        response.raise_for_status()
        return response.text[:8000]  # Limit content length
    except Exception as e:
        return f"An error occurred during scraping: {e}"

async def read_resume(file_path: str = "./fake_resume.md") -> str:
    """Read resume file content."""
    print(f"--- Using Tool: read_resume from file: '{file_path}' ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"An error occurred reading resume: {e}"

async def search_resume_content(query: str) -> str:
    """Search through resume content for specific information."""
    print(f"--- Using Tool: search_resume_content with query: '{query}' ---")
    try:
        resume_content = await read_resume()
        # Simple keyword search in resume content
        lines = resume_content.split('\n')
        relevant_lines = [line for line in lines if query.lower() in line.lower()]
        return '\n'.join(relevant_lines[:10])  # Return top 10 matching lines
    except Exception as e:
        return f"An error occurred searching resume: {e}"

# -----------------------------
# Create Tool Instances
# -----------------------------
search_tool = FunctionTool(func=search_internet)
scrape_tool = FunctionTool(func=scrape_website)
resume_read_tool = FunctionTool(func=read_resume)
resume_search_tool = FunctionTool(func=search_resume_content)

# -----------------------------
# Pydantic Models for Outputs
# -----------------------------
class JobRequirements(BaseModel):
    """Structured job requirements from job posting analysis."""
    title: str = Field(description="Job title")
    required_skills: list = Field(description="List of required technical skills")
    qualifications: list = Field(description="List of required qualifications")
    experience_level: str = Field(description="Required experience level")
    responsibilities: list = Field(description="Main job responsibilities")

class CandidateProfile(BaseModel):
    """Comprehensive candidate profile."""
    name: str = Field(description="Candidate name")
    summary: str = Field(description="Professional summary")
    skills: list = Field(description="Technical and soft skills")
    experience: list = Field(description="Work experience details")
    projects: list = Field(description="Notable projects")
    education: str = Field(description="Educational background")

class TailoredResume(BaseModel):
    """Tailored resume output."""
    content: str = Field(description="Complete tailored resume in markdown format")

class InterviewMaterials(BaseModel):
    """Interview preparation materials."""
    questions: list = Field(description="Potential interview questions")
    talking_points: list = Field(description="Key talking points for candidate")
    preparation_notes: str = Field(description="Additional preparation guidance")

# =============================================================================
# AGENT 1: TECH JOB RESEARCHER
# =============================================================================
tech_job_researcher = LlmAgent(
    name="tech_job_researcher",
    model=AGENT_MODEL,
    description="Specialized in analyzing job postings to extract key requirements and qualifications.",
    
    instruction="""
    ## PRIMARY ROLE
    You are a Tech Job Researcher with expertise in analyzing job postings to help job applicants understand requirements.
    
    ## CORE RESPONSIBILITIES & TASKS
    1. **Job Posting Analysis Task**
       - Analyze the provided job posting URL to extract key information
       - Use `search_internet` and `scrape_website` tools extensively
       - Extract key skills, experiences, and qualifications required
       - Identify and categorize the requirements into technical and soft skills
       - Determine experience level expectations and responsibilities
    
    ## RESEARCH METHODOLOGY
    1. Scrape the job posting URL provided in the input
    2. Search for additional information about the company and role
    3. Analyze the job description for explicit and implicit requirements
    4. Categorize findings into structured format
    
    ## EXPECTED OUTPUT FORMAT (Markdown)
    ### Job Analysis Summary
    - **Job Title**: [Extracted job title]
    - **Company**: [Company name and brief description]
    - **Location**: [Job location/remote options]
    
    ### Required Skills & Qualifications
    - **Technical Skills**: List all programming languages, frameworks, tools
    - **Soft Skills**: Communication, leadership, teamwork requirements  
    - **Experience Level**: Years of experience required
    - **Education**: Degree requirements and preferences
    - **Certifications**: Any specific certifications mentioned
    
    ### Key Responsibilities
    - List main job responsibilities and daily tasks
    - Identify team structure and collaboration requirements
    - Note any special requirements or unique aspects
    
    ### Company Culture & Values
    - Extract information about company culture
    - Identify what the company values in employees
    - Note any specific company initiatives or goals
    
    ## QUALITY STANDARDS
    - Extract ALL explicitly mentioned requirements
    - Identify implicit requirements from job description context
    - Provide specific, actionable insights for job applicants
    - Use clear, structured formatting for easy reference
    """,
    
    tools=[search_tool, scrape_tool],
    output_key="job_requirements"
)

# =============================================================================
# AGENT 2: PERSONAL PROFILER FOR ENGINEERS
# =============================================================================
personal_profiler = LlmAgent(
    name="personal_profiler",
    model=AGENT_MODEL,
    description="Expert at creating comprehensive personal and professional profiles from multiple sources.",
    
    instruction="""
    ## PRIMARY ROLE
    You are a Personal Profiler for Engineers specializing in creating comprehensive candidate profiles.
    
    ## CORE RESPONSIBILITIES & TASKS
    1. **Comprehensive Profile Compilation Task**
       - Use GitHub URL, personal writeup, and resume to build complete profile
       - Utilize `scrape_website`, `search_internet`, `read_resume`, and `search_resume_content` tools
       - Extract and synthesize information from diverse sources
       - Create detailed personal and professional profile
       - Analyze coding style, project contributions, and technical expertise
    
    ## RESEARCH METHODOLOGY
    1. Scrape GitHub profile to understand technical skills and projects
    2. Read and analyze resume for structured experience information
    3. Process personal writeup for additional context and personality
    4. Search for additional public information about the candidate
    5. Synthesize all information into comprehensive profile
    
    ## EXPECTED OUTPUT FORMAT (Markdown)
    ### Personal & Professional Profile
    
    #### Professional Summary
    - Comprehensive overview combining all sources
    - Years of experience and specialization areas
    - Leadership and team experience
    - Career progression and achievements
    
    #### Technical Skills & Expertise
    - Programming languages with proficiency levels
    - Frameworks and technologies used
    - Tools and development environments
    - Architecture and system design experience
    
    #### Project Portfolio Analysis
    - Notable projects from GitHub and resume
    - Technologies used in each project
    - Problem-solving approaches demonstrated
    - Scale and complexity of projects handled
    
    #### Professional Experience
    - Detailed work history from resume
    - Key achievements and responsibilities
    - Team sizes and leadership roles
    - Industry experience and domain knowledge
    
    #### Communication Style & Soft Skills
    - Writing style from GitHub and personal writeup
    - Collaboration patterns from project history
    - Problem-solving approach
    - Learning agility and adaptability
    
    #### Education & Continuous Learning
    - Formal education background
    - Certifications and additional training
    - Self-directed learning evidence
    - Contribution to open-source communities
    
    ## QUALITY STANDARDS
    - Synthesize information from ALL provided sources
    - Highlight unique strengths and differentiators
    - Identify areas of expertise and specialization
    - Provide evidence-based insights about capabilities
    - Create actionable profile for resume tailoring
    """,
    
    tools=[scrape_tool, search_tool, resume_read_tool, resume_search_tool],
    output_key="candidate_profile"
)

# =============================================================================
# AGENT 3: RESUME STRATEGIST FOR ENGINEERS
# =============================================================================
resume_strategist = LlmAgent(
    name="resume_strategist",
    model=AGENT_MODEL,
    description="Specialized in tailoring resumes to highlight relevant qualifications for specific job requirements.",
    
    instruction="""
    ## PRIMARY ROLE
    You are a Resume Strategist for Engineers focused on optimizing resumes for job market success.
    
    ## CORE RESPONSIBILITIES & TASKS
    1. **Resume Tailoring & Enhancement Task**
       - Use job requirements from researcher and candidate profile from profiler
       - Employ `read_resume` and `search_resume_content` tools to work with current resume
       - Tailor resume to highlight most relevant areas for the specific job
       - Update every section: summary, experience, skills, education
       - Ensure resume effectively showcases qualifications matching job requirements
       - DO NOT fabricate information - only enhance existing content
    
    ## TAILORING METHODOLOGY
    1. Analyze job requirements against candidate profile
    2. Identify key matches and alignment opportunities  
    3. Prioritize most relevant experiences and skills
    4. Restructure content to lead with strongest matches
    5. Quantify achievements where possible
    6. Ensure ATS-friendly formatting and keyword optimization
    
    ## EXPECTED OUTPUT FORMAT (Complete Tailored Resume in Markdown)
    
    # [CANDIDATE NAME]
    **[Phone] | [Email] | [Location] | [LinkedIn] | [GitHub]**
    
    ## PROFESSIONAL SUMMARY
    [Tailored 3-4 line summary emphasizing alignment with job requirements]
    
    ## TECHNICAL SKILLS
    - **Programming Languages**: [Prioritized based on job requirements]
    - **Frameworks & Technologies**: [Relevant to the position]
    - **Tools & Platforms**: [Matching job needs]
    - **Specializations**: [Aligned with role requirements]
    
    ## PROFESSIONAL EXPERIENCE
    
    ### [Job Title] | [Company] | [Dates]
    - [Achievement quantified with metrics relevant to job requirements]
    - [Responsibility highlighting skills mentioned in job posting]
    - [Project outcome demonstrating required capabilities]
    
    ### [Previous roles following same pattern...]
    
    ## KEY PROJECTS
    
    ### [Project Name] | [Technology Stack]
    - [Brief description emphasizing relevant technologies and outcomes]
    - [Quantified results where applicable]
    
    ## EDUCATION
    **[Degree]** | [Institution] | [Year]
    - [Relevant coursework, GPA if strong, honors]
    
    ## CERTIFICATIONS & ADDITIONAL TRAINING
    - [Relevant certifications matching job requirements]
    
    ## QUALITY STANDARDS FOR TAILORED RESUME
    - Lead with information most relevant to the specific job
    - Use keywords from job posting naturally throughout
    - Quantify achievements with specific metrics
    - Demonstrate progression and growth
    - Show direct alignment between experience and job requirements
    - Maintain professional formatting and ATS compatibility
    - Keep to 1-2 pages maximum length
    - Ensure every bullet point adds value and relevance
    """,
    
    tools=[resume_read_tool, resume_search_tool],
    output_key="tailored_resume"
)

# =============================================================================
# AGENT 4: ENGINEERING INTERVIEW PREPARER
# =============================================================================
interview_preparer = LlmAgent(
    name="interview_preparer",
    model=AGENT_MODEL,
    description="Creates comprehensive interview preparation materials based on job requirements and tailored resume.",
    
    instruction="""
    ## PRIMARY ROLE
    You are an Engineering Interview Preparer specializing in creating targeted interview preparation materials.
    
    ## CORE RESPONSIBILITIES & TASKS
    1. **Interview Materials Development Task**
       - Use job requirements, candidate profile, and tailored resume from previous agents
       - Create comprehensive set of potential interview questions
       - Develop talking points based on resume and job requirements
       - Generate relevant technical questions and discussion points
       - Ensure materials help candidate highlight main resume points
       - Focus on demonstrating alignment with job requirements
    
    ## PREPARATION METHODOLOGY
    1. Analyze job requirements for likely interview topics
    2. Review tailored resume for key talking points
    3. Anticipate behavioral and technical questions
    4. Create response frameworks highlighting relevant experience
    5. Develop questions candidate should ask interviewer
    6. Prepare examples demonstrating required skills
    
    ## EXPECTED OUTPUT FORMAT (Markdown)
    
    # Interview Preparation Materials
    
    ## Executive Summary
    **Position**: [Job Title at Company]
    **Key Focus Areas**: [Main areas to emphasize based on job requirements]
    **Interview Strategy**: [Overall approach for this specific role]
    
    ## Core Talking Points
    ### Technical Expertise Highlights
    - **[Skill/Technology 1]**: [Specific example from experience with quantified impact]
    - **[Skill/Technology 2]**: [Project or achievement demonstrating this skill]
    - **[Skill/Technology 3]**: [How this aligns with job requirements]
    
    ### Professional Achievement Stories
    1. **[Achievement Title]**: [STAR method response - Situation, Task, Action, Result]
    2. **[Leadership Example]**: [Specific example of leadership or teamwork]
    3. **[Problem-Solving Example]**: [Technical challenge overcome with measurable results]
    
    ## Anticipated Interview Questions & Suggested Responses
    
    ### Technical Questions
    1. **Q**: [Likely technical question based on job requirements]
       **A**: [Response framework highlighting relevant experience]
    
    2. **Q**: [System design or architecture question]
       **A**: [Approach mentioning specific technologies from resume]
    
    3. **Q**: [Coding/development process question]
       **A**: [Response showcasing methodology and tools experience]
    
    ### Behavioral Questions
    1. **Q**: "Tell me about a challenging project you've worked on."
       **A**: [Framework using specific project from resume/profile]
    
    2. **Q**: "How do you handle working in a team environment?"
       **A**: [Examples from experience emphasizing collaboration skills]
    
    3. **Q**: "Describe a time you had to learn a new technology quickly."
       **A**: [Learning agility example relevant to job requirements]
    
    ### Company-Specific Questions
    1. **Q**: [Question about company culture/values]
       **A**: [Response showing research and alignment]
    
    2. **Q**: "Why are you interested in this role?"
       **A**: [Tailored response connecting background to opportunity]
    
    ## Questions to Ask the Interviewer
    - [Thoughtful questions about role, team, technology stack]
    - [Questions about growth opportunities and company direction]
    - [Technical questions about architecture, processes, challenges]
    
    ## Final Interview Reminders
    - **Key Strengths to Emphasize**: [Top 3-5 selling points]
    - **Stories to Tell**: [Most compelling examples to share]
    - **Technical Demos**: [Projects or code samples to discuss]
    - **Follow-up Strategy**: [Post-interview action plan]
    
    ## QUALITY STANDARDS
    - All questions should be realistic and likely for the role
    - Responses should be specific to candidate's background
    - Technical questions should match job requirements complexity
    - Behavioral questions should allow highlighting of key experiences
    - Materials should build confidence while being realistic
    - Include both preparation and strategy elements
    """,
    
    tools=[resume_read_tool, resume_search_tool],
    output_key="interview_materials"
)

# =============================================================================
# SEQUENTIAL WORKFLOW
# =============================================================================
job_application_agent = SequentialAgent(
    name='JobApplicationAgent',
    description="""
    Complete job application preparation workflow:
    1. Tech Job Researcher - Analyze job posting and extract requirements
    2. Personal Profiler - Create comprehensive candidate profile from multiple sources
    3. Resume Strategist - Tailor resume to highlight relevant qualifications
    4. Interview Preparer - Create targeted interview preparation materials
    
    This system helps job applicants create tailored applications that effectively
    demonstrate their qualifications for specific positions.
    """,
    sub_agents=[
        tech_job_researcher,
        personal_profiler,
        resume_strategist,
        interview_preparer
    ],
)

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================
async def run_job_application_workflow(job_posting_url: str, github_url: str, personal_writeup: str):
    """Run the complete job application preparation workflow."""
    
    session_service = InMemorySessionService()
    print("🚀 Starting Job Application Preparation Workflow...")
    
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=job_application_agent, app_name=APP_NAME, session_service=session_service)
    
    # Format the input query
    query = f"""
    Job Application Preparation Request:
    
    Job Posting URL: {job_posting_url}
    GitHub Profile: {github_url}
    
    Personal Background:
    {personal_writeup}
    
    Please analyze the job requirements, create a comprehensive profile, 
    tailor the resume, and prepare interview materials for this position.
    """
    
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    try:
        final_results = {}
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            print(f"{'='*80}")
            print(f"Agent: {event.author}")
            print(f"{'='*80}")
            
            if event.is_final_response():
                agent_name = event.author
                content_text = event.content.parts[0].text
                final_results[agent_name] = content_text
                
                # Save specific outputs to files
                if agent_name == "resume_strategist":
                    with open("tailored_resume.md", "w", encoding='utf-8') as f:
                        f.write(content_text)
                    print("✅ Tailored resume saved to tailored_resume.md")
                
                elif agent_name == "interview_preparer":
                    with open("interview_materials.md", "w", encoding='utf-8') as f:
                        f.write(content_text)
                    print("✅ Interview materials saved to interview_materials.md")
                    break  # Final agent completed
    
    except Exception as e:
        print(f"❌ Error during workflow execution: {e}")
        return None
    
    print("\n🎉 Job Application Preparation Completed Successfully!")
    return final_results

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Example inputs (similar to original CrewAI version)
    job_inputs = {
        'job_posting_url': 'https://www.linkedin.com/jobs/collections/recommended/?currentJobId=4284779147&origin=JYMBII_IN_APP_NOTIFICATION&originToLandingJobPostings=4293600008%2C4295007000',
        'github_url': 'https://github.com/tariqjamil-bwp',
        'personal_writeup': """Tariq J. is an accomplished Software Engineering Leader with 5 years of experience, 
        specializing in managing AI/ML abnd Agentic application development. Ideal for leadership 
        roles that require a strategic and innovative approach."""
    }
    
    try:
        results = asyncio.run(run_job_application_workflow(
            job_inputs['job_posting_url'],
            job_inputs['github_url'], 
            job_inputs['personal_writeup']
        ))
        
        if results:
            print("\n📁 Generated Files:")
            print("- tailored_resume.md")
            print("- interview_materials.md")
            
    except KeyboardInterrupt:
        print("\n❌ Workflow interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")