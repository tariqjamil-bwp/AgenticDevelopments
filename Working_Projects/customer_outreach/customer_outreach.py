# -*- coding: utf-8 -*-
"""
Google ADK Customer Outreach Multi-Agent System
Using CrewAI Tools directly in Google ADK for better functionality
Token optimized with proper file reading and chunking capabilities
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
from typing import Optional, List, Dict

# Import CrewAI Tools directly
from crewai_tools import DirectoryReadTool, FileReadTool, SerperDevTool
from crewai.tools import BaseTool

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# -----------------------------
# Constants and API Key Checks
# -----------------------------
APP_NAME = "customer_outreach_app"
USER_ID = "sales_user_001"
SESSION_ID = "outreach_session_001"

if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY environment variable not set.")
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# -----------------------------
# LLM Model
# -----------------------------
AGENT_MODEL = LiteLlm("gemini/gemini-2.5-flash")

# -----------------------------
# Initialize CrewAI Tools
# -----------------------------
# Create instructions directory if it doesn't exist
instructions_dir = "./instructions"
os.makedirs(instructions_dir, exist_ok=True)

# Create sample instruction files if they don't exist
sample_files = {
    "company_research_guidelines.txt": """
Company Research Guidelines:
1. Focus on recent news and developments (last 6 months)
2. Identify key decision makers and their backgrounds
3. Look for technology stack and current solutions
4. Find pain points and growth opportunities
5. Check for recent funding, partnerships, or product launches
6. Analyze company culture and values from their content
7. Identify competitive landscape and market position
""",
    "outreach_best_practices.txt": """
Outreach Best Practices:
1. Personalize subject lines with company name or recent news
2. Keep emails under 150 words for better response rates
3. Reference specific company achievements or milestones
4. Focus on business outcomes, not just product features
5. Include clear, low-commitment call-to-action
6. Use social proof and relevant case studies
7. Follow up within 3-5 business days if no response
8. Provide multiple ways to connect (email, LinkedIn, phone)
""",
    "ideal_customer_profile.txt": """
Ideal Customer Profile:
- Company Size: 50-5000 employees
- Industry: Technology, Education, Healthcare, Financial Services
- Revenue: $10M-$1B annual revenue
- Technology Focus: AI/ML adoption, Digital transformation
- Pain Points: Data analytics, Process automation, Training needs
- Decision Makers: CTO, CEO, VP Engineering, Head of Data
- Buying Signals: Recent funding, team expansion, new product launches
- Geographic: North America, Europe, Asia-Pacific (English-speaking markets)
"""
}

for filename, content in sample_files.items():
    filepath = os.path.join(instructions_dir, filename)
    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

# Initialize CrewAI tools
directory_read_tool = DirectoryReadTool(directory=instructions_dir)
file_read_tool = FileReadTool()
serper_search_tool = SerperDevTool()

# -----------------------------
# Custom Sentiment Analysis Tool (as in original CrewAI example)
# -----------------------------
class SentimentAnalysisTool(BaseTool):
    name: str = "Sentiment Analysis Tool"
    description: str = ("Analyzes the sentiment of text "
                       "to ensure positive and engaging communication.")

    def _run(self, text: str) -> str:
        """
        Enhanced sentiment analysis with keyword matching
        Returns: positive, negative, or neutral
        """
        positive_words = [
            'excellent', 'outstanding', 'impressive', 'successful', 'innovative',
            'growth', 'expansion', 'achievement', 'milestone', 'breakthrough',
            'leading', 'pioneer', 'award', 'recognition', 'partnership',
            'collaboration', 'opportunity', 'potential', 'exciting', 'remarkable'
        ]
        
        negative_words = [
            'challenging', 'difficult', 'problem', 'issue', 'decline', 'loss',
            'failure', 'setback', 'concern', 'risk', 'threat', 'crisis',
            'downturn', 'struggling', 'disappointing', 'underperforming'
        ]
        
        neutral_words = [
            'stable', 'consistent', 'standard', 'typical', 'regular',
            'normal', 'average', 'moderate', 'steady', 'maintaining'
        ]
        
        text_lower = text.lower()
        positive_score = sum(2 if word in text_lower else 0 for word in positive_words)
        negative_score = sum(2 if word in text_lower else 0 for word in negative_words)
        neutral_score = sum(1 if word in text_lower else 0 for word in neutral_words)
        
        # Determine sentiment based on scores
        if positive_score > negative_score and positive_score > neutral_score:
            return "positive"
        elif negative_score > positive_score and negative_score > neutral_score:
            return "negative"
        else:
            return "neutral"

sentiment_analysis_tool = SentimentAnalysisTool()

# -----------------------------
# Wrapper Functions for Google ADK Integration
# -----------------------------
async def search_company_async(query: str) -> str:
    """Async wrapper for SerperDevTool"""
    try:
        print(f"🔍 Searching: {query}")
        result = serper_search_tool._run(query)
        # Limit result length to optimize tokens
        if len(str(result)) > 2000:
            return str(result)[:2000] + "... [truncated for token efficiency]"
        return str(result)
    except Exception as e:
        return f"Search error: {str(e)}"

async def read_directory_async(directory_path: str = instructions_dir) -> str:
    """Async wrapper for DirectoryReadTool"""
    try:
        print(f"📁 Reading directory: {directory_path}")
        result = directory_read_tool._run(directory_path)
        return str(result)
    except Exception as e:
        return f"Directory read error: {str(e)}"

async def read_file_async(file_path: str) -> str:
    """Async wrapper for FileReadTool"""
    try:
        print(f"📄 Reading file: {file_path}")
        result = file_read_tool._run(file_path)
        # Limit file content to optimize tokens
        if len(str(result)) > 1500:
            return str(result)[:1500] + "... [truncated for token efficiency]"
        return str(result)
    except Exception as e:
        return f"File read error: {str(e)}"

async def analyze_sentiment_async(text: str) -> str:
    """Async wrapper for SentimentAnalysisTool"""
    try:
        print(f"📊 Analyzing sentiment for text: {text[:50]}...")
        result = sentiment_analysis_tool._run(text)
        return str(result)
    except Exception as e:
        return f"Sentiment analysis error: {str(e)}"

# -----------------------------
# Create Google ADK Function Tools
# -----------------------------
search_tool = FunctionTool(func=search_company_async)
directory_tool = FunctionTool(func=read_directory_async)
file_tool = FunctionTool(func=read_file_async)
sentiment_tool = FunctionTool(func=analyze_sentiment_async)

# -----------------------------
# Pydantic Models for Structured Outputs
# -----------------------------
class LeadProfile(BaseModel):
    """Comprehensive lead profile structure."""
    company_name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    company_size: Optional[str] = Field(description="Employee count or revenue range")
    key_personnel: List[Dict[str, str]] = Field(description="Key decision makers with titles")
    recent_developments: List[str] = Field(description="Recent milestones, funding, or news")
    technology_stack: Optional[List[str]] = Field(description="Known technologies or platforms")
    potential_needs: List[str] = Field(description="Identified business needs or pain points")
    competitive_position: Optional[str] = Field(description="Market position and competitors")
    engagement_strategy: str = Field(description="Recommended outreach approach")
    lead_score: int = Field(description="Lead quality score 1-10", ge=1, le=10)

class OutreachCampaign(BaseModel):
    """Structured outreach campaign with multiple touchpoints."""
    target_person: str = Field(description="Primary contact person and title")
    campaign_theme: str = Field(description="Overall campaign messaging theme")
    email_sequence: List[Dict[str, str]] = Field(description="Sequential email campaign")
    subject_lines: List[str] = Field(description="A/B test subject line variants")
    key_value_propositions: List[str] = Field(description="Main value points to highlight")
    call_to_action: str = Field(description="Primary desired action")
    follow_up_timeline: str = Field(description="Follow-up schedule and approach")
    success_metrics: List[str] = Field(description="Campaign success measurement criteria")

# =============================================================================
# AGENT 1: LEAD RESEARCH AGENT (Enhanced with CrewAI Tools)
# =============================================================================
lead_research_agent = LlmAgent(
    name="lead_research_agent",
    model=AGENT_MODEL,
    description="Advanced lead research specialist using comprehensive data sources and analysis tools.",
    
    instruction="""
    ## PRIMARY ROLE
    You are a Lead Research Agent specializing in comprehensive lead analysis using multiple data sources.
    
    ## AVAILABLE TOOLS & DATA SOURCES
    1. **search_company_async**: Web search for recent company information, news, and developments
    2. **read_directory_async**: Access research guidelines and best practices from instructions directory
    3. **read_file_async**: Read specific files containing company data, guidelines, or customer profiles
    4. **analyze_sentiment_async**: Analyze sentiment of company communications and content
    
    ## RESEARCH METHODOLOGY
    
    ### Phase 1: Initial Research Setup
    1. Read research guidelines from instructions directory
    2. Review ideal customer profile criteria
    3. Establish research framework based on best practices
    
    ### Phase 2: Company Intelligence Gathering
    1. Search for recent company news, press releases, and developments
    2. Identify key personnel, especially decision-makers in target roles
    3. Research company's technology stack and current solutions
    4. Analyze recent funding, partnerships, or strategic initiatives
    5. Evaluate company culture and values from public communications
    
    ### Phase 3: Needs Analysis & Lead Scoring
    1. Identify potential business needs and pain points
    2. Assess fit with our ideal customer profile
    3. Analyze competitive landscape and market position
    4. Score lead quality based on multiple criteria (1-10 scale)
    5. Develop preliminary engagement strategy
    
    ## LEAD SCORING CRITERIA (1-10 scale)
    - **Fit with ICP**: 0-2 points (company size, industry, revenue)
    - **Decision Maker Access**: 0-2 points (identifiable contacts, roles)
    - **Buying Signals**: 0-2 points (recent funding, growth, initiatives)
    - **Technology Alignment**: 0-2 points (current tech stack, needs)
    - **Timing Indicators**: 0-2 points (recent milestones, market timing)
    
    ## OUTPUT STRUCTURE
    
    ### Company Overview
    - Company name, industry, and size
    - Core business model and market position
    - Recent significant developments (last 6 months)
    
    ### Key Personnel Analysis  
    - Primary decision makers with full names and titles
    - Background information where available
    - Potential influencers and technical contacts
    
    ### Business Intelligence
    - Current technology stack and solutions
    - Recent strategic initiatives or challenges
    - Competitive landscape and positioning
    
    ### Needs Assessment
    - Identified pain points or opportunities
    - Potential areas for our solution impact
    - Budget and decision-making indicators
    
    ### Lead Qualification
    - Overall lead score (1-10) with justification
    - Recommended engagement approach
    - Priority level and timing recommendations
    
    ## RESEARCH QUALITY STANDARDS
    - Verify information accuracy across multiple sources
    - Focus on recent developments (last 6-12 months)
    - Prioritize information from official company sources
    - Include specific names, dates, and quantifiable data
    - Avoid speculation - clearly distinguish facts from assumptions
    - Provide source attribution where possible
    
    ## TOKEN OPTIMIZATION
    - Focus on high-value information relevant to sales process
    - Summarize lengthy content while preserving key insights
    - Prioritize actionable intelligence over general information
    - Use structured format for easy consumption by outreach agent
    """,
    
    tools=[search_tool, directory_tool, file_tool, sentiment_tool],
    output_key="lead_profile"
)

# =============================================================================
# AGENT 2: CAMPAIGN STRATEGIST AGENT (Enhanced with CrewAI Tools)
# =============================================================================
campaign_strategist_agent = LlmAgent(
    name="campaign_strategist_agent",
    model=AGENT_MODEL,
    description="Expert campaign strategist creating multi-touch outreach sequences with high conversion potential.",
    
    instruction="""
    ## PRIMARY ROLE
    You are a Campaign Strategist Agent specializing in multi-touch outreach campaigns that convert leads into opportunities.
    
    ## AVAILABLE TOOLS & CAMPAIGN RESOURCES
    1. **search_company_async**: Additional research for campaign personalization
    2. **read_directory_async**: Access outreach best practices and templates
    3. **read_file_async**: Read specific campaign guidelines and customer success stories
    4. **analyze_sentiment_async**: Ensure all campaign messaging maintains positive, engaging tone
    
    ## CAMPAIGN DEVELOPMENT PROCESS
    
    ### Phase 1: Campaign Strategy Foundation
    1. Read outreach best practices from instructions directory
    2. Review lead profile and research findings
    3. Identify primary campaign theme and messaging angles
    4. Define success metrics and conversion goals
    
    ### Phase 2: Message Crafting & Personalization
    1. Develop compelling subject lines with A/B test variants
    2. Create personalized email sequence (3-5 touches)
    3. Incorporate specific company achievements and developments
    4. Ensure all messaging passes positive sentiment analysis
    
    ### Phase 3: Multi-Channel Strategy Development
    1. Design follow-up sequence with varied approaches
    2. Include social proof and relevant case studies
    3. Create multiple call-to-action options
    4. Plan alternative engagement channels (LinkedIn, phone)
    
    ## EMAIL SEQUENCE STRUCTURE
    
    ### Email 1: Initial Outreach (within 24 hours)
    **Purpose**: Establish connection and credibility
    - Personalized subject line referencing recent company news
    - Congratulate on specific achievement or milestone
    - Brief value proposition with relevant case study
    - Soft call-to-action (industry insights, brief conversation)
    
    ### Email 2: Value-Add Follow-up (3-5 days later)
    **Purpose**: Provide value without asking for anything
    - Share relevant industry insight or resource
    - Reference additional company research findings
    - Include customer success story from similar company
    - Low-pressure engagement offer
    
    ### Email 3: Strategic Follow-up (7-10 days after Email 2)
    **Purpose**: Create urgency while maintaining relationship
    - Reference specific business challenge or opportunity
    - Present clear ROI or business impact proposition
    - Include specific demo or consultation offer
    - Provide multiple engagement options
    
    ### Email 4: Alternative Angle (14 days after Email 3)
    **Purpose**: Try different messaging approach if no response
    - Approach from different business angle or stakeholder perspective
    - Include more detailed case study or ROI calculator
    - Offer educational webinar or industry report
    - Final attempt with "permission to remove" option
    
    ## PERSONALIZATION ELEMENTS
    
    ### Company-Specific Details
    - Recent news, funding, or product launches
    - Industry challenges and market trends
    - Competitive landscape positioning
    - Technology stack and integration opportunities
    
    ### Individual Personalization  
    - Decision maker's professional background
    - Recent LinkedIn activity or company posts
    - Industry conference speaking or content creation
    - Professional interests and expertise areas
    
    ## CAMPAIGN SUCCESS METRICS
    - Email open rates (target: 25-35%)
    - Click-through rates (target: 3-8%)  
    - Response rates (target: 5-15%)
    - Meeting conversion rates (target: 20-40% of responses)
    - Pipeline progression (target: 30-50% to next stage)
    
    ## MESSAGE TONE & QUALITY STANDARDS
    - Professional yet conversational tone
    - Confident without being pushy or aggressive
    - Genuine interest in prospect's business success
    - All messages must pass sentiment analysis as "positive"
    - Clear, specific value propositions with quantifiable benefits
    - Error-free grammar and professional formatting
    
    ## CAMPAIGN OPTIMIZATION
    - A/B test subject lines and key message elements
    - Track engagement metrics for continuous improvement
    - Provide alternative messaging for different personas
    - Include social proof and third-party validation
    - Ensure mobile-friendly formatting for all messages
    
    ## EXPECTED OUTPUT FORMAT
    
    ### Campaign Overview
    - Target persona and primary messaging theme
    - Campaign timeline and touch sequence
    - Key value propositions and differentiation points
    
    ### Email Sequence Templates
    - Complete email templates for each touch
    - Multiple subject line variants for A/B testing
    - Personalization merge fields clearly marked
    
    ### Alternative Engagement Strategies
    - LinkedIn outreach messages
    - Phone calling scripts and objection handling
    - Follow-up sequences for different response scenarios
    
    ### Success Tracking Framework
    - Key performance indicators and targets
    - Campaign analytics and optimization recommendations
    - Next steps for qualified leads
    """,
    
    tools=[search_tool, directory_tool, file_tool, sentiment_tool],
    output_key="outreach_campaign"
)

# =============================================================================
# SEQUENTIAL WORKFLOW
# =============================================================================
customer_outreach_system = SequentialAgent(
    name='CustomerOutreachSystem',
    description="""
    Advanced customer outreach system using CrewAI tools integration:
    1. Lead Research Agent - Comprehensive lead analysis using multiple data sources
    2. Campaign Strategist Agent - Multi-touch outreach campaign development
    
    Features:
    - CrewAI tools integration for enhanced data access
    - File-based research guidelines and best practices
    - Multi-channel campaign strategies
    - Token-optimized processing with quality controls
    """,
    sub_agents=[
        lead_research_agent,
        campaign_strategist_agent
    ],
)

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================
async def run_comprehensive_outreach(campaign_inputs: dict):
    """Run comprehensive outreach campaign with CrewAI tools integration."""
    
    session_service = InMemorySessionService()
    print("🚀 Starting Comprehensive Customer Outreach Campaign...")
    
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=customer_outreach_system, app_name=APP_NAME, session_service=session_service)
    
    # Enhanced input query with research depth
    query = f"""
    Comprehensive Customer Outreach Campaign:
    
    Target Company: {campaign_inputs['lead_name']}
    Industry Sector: {campaign_inputs['industry']}
    Primary Contact: {campaign_inputs['key_decision_maker']} ({campaign_inputs['position']})
    Recent Development: {campaign_inputs['milestone']}
    Campaign Objective: {campaign_inputs.get('objective', 'Generate qualified sales opportunity')}
    
    Please conduct thorough lead research using all available data sources and create 
    a comprehensive multi-touch outreach campaign optimized for high conversion rates.
    
    Research Focus Areas:
    1. Company background, size, and market position
    2. Key decision makers and influencers
    3. Recent business developments and initiatives
    4. Technology stack and current solution gaps
    5. Competitive landscape and differentiation opportunities
    
    Campaign Requirements:
    1. Multi-touch email sequence (3-4 emails)
    2. Personalized messaging based on research findings
    3. A/B test subject line variants
    4. Alternative engagement strategies
    5. Success metrics and tracking framework
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
                
                # Save detailed outputs
                company_safe_name = campaign_inputs['lead_name'].replace(' ', '_').replace('.', '').lower()
                
                if agent_name == "lead_research_agent":
                    filename = f"lead_research_{company_safe_name}.md"
                    with open(filename, "w", encoding='utf-8') as f:
                        f.write(f"# Comprehensive Lead Research: {campaign_inputs['lead_name']}\n\n")
                        f.write(f"**Research Date**: {asyncio.get_event_loop().time()}\n")
                        f.write(f"**Target Contact**: {campaign_inputs['key_decision_maker']} ({campaign_inputs['position']})\n")
                        f.write(f"**Industry**: {campaign_inputs['industry']}\n\n")
                        f.write("---\n\n")
                        f.write(content_text)
                    print(f"✅ Lead research saved: {filename}")
                
                elif agent_name == "campaign_strategist_agent":
                    filename = f"campaign_strategy_{company_safe_name}.md"
                    with open(filename, "w", encoding='utf-8') as f:
                        f.write(f"# Outreach Campaign Strategy: {campaign_inputs['lead_name']}\n\n")
                        f.write(f"**Campaign Target**: {campaign_inputs['key_decision_maker']} ({campaign_inputs['position']})\n")
                        f.write(f"**Campaign Theme**: Based on {campaign_inputs['milestone']}\n")
                        f.write(f"**Industry Context**: {campaign_inputs['industry']}\n\n")
                        f.write("---\n\n")
                        f.write(content_text)
                    print(f"✅ Campaign strategy saved: {filename}")
                    break  # Final agent completed
    
    except Exception as e:
        print(f"❌ Error during campaign execution: {e}")
        return None
    
    print("\n🎉 Comprehensive Customer Outreach Campaign Completed!")
    return final_results

# =============================================================================
# BATCH PROCESSING WITH ENHANCED REPORTING
# =============================================================================
async def run_batch_outreach_campaigns(campaign_list: List[dict], batch_name: str = "batch_campaign"):
    """Run multiple campaigns with comprehensive reporting."""
    
    print(f"🚀 Starting {batch_name} with {len(campaign_list)} target companies...")
    
    batch_results = {}
    successful_campaigns = 0
    failed_campaigns = 0
    
    for i, campaign_inputs in enumerate(campaign_list, 1):
        print(f"\n{'='*100}")
        print(f"PROCESSING CAMPAIGN {i}/{len(campaign_list)}: {campaign_inputs['lead_name']}")
        print(f"{'='*100}")
        
        try:
            results = await run_comprehensive_outreach(campaign_inputs)
            if results:
                batch_results[campaign_inputs['lead_name']] = {
                    'status': 'success',
                    'results': results,
                    'target_contact': campaign_inputs['key_decision_maker'],
                    'industry': campaign_inputs['industry']
                }
                successful_campaigns += 1
            else:
                batch_results[campaign_inputs['lead_name']] = {
                    'status': 'failed',
                    'error': 'No results returned'
                }
                failed_campaigns += 1
                
            # Brief pause between campaigns
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"❌ Campaign failed for {campaign_inputs['lead_name']}: {e}")
            batch_results[campaign_inputs['lead_name']] = {
                'status': 'failed',
                'error': str(e)
            }
            failed_campaigns += 1
            continue
    
    # Generate comprehensive batch report
    batch_report_filename = f"{batch_name}_report.md"
    with open(batch_report_filename, "w", encoding='utf-8') as f:
        f.write(f"# {batch_name.replace('_', ' ').title()} Report\n\n")
        f.write(f"**Execution Date**: {asyncio.get_event_loop().time()}\n")
        f.write(f"**Total Campaigns**: {len(campaign_list)}\n")
        f.write(f"**Successful**: {successful_campaigns}\n")
        f.write(f"**Failed**: {failed_campaigns}\n")
        f.write(f"**Success Rate**: {(successful_campaigns/len(campaign_list)*100):.1f}%\n\n")
        
        f.write("## Campaign Results Summary\n\n")
        for company, result in batch_results.items():
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            f.write(f"{status_emoji} **{company}**")
            if result['status'] == 'success':
                f.write(f" - Target: {result['target_contact']} | Industry: {result['industry']}\n")
            else:
                f.write(f" - Error: {result['error']}\n")
        
        f.write("\n## Generated Files\n\n")
        for company, result in batch_results.items():
            if result['status'] == 'success':
                company_safe = company.replace(' ', '_').replace('.', '').lower()
                f.write(f"- `lead_research_{company_safe}.md`\n")
                f.write(f"- `campaign_strategy_{company_safe}.md`\n")
    
    print(f"\n🎉 Batch campaign completed!")
    print(f"📊 Results: {successful_campaigns}/{len(campaign_list)} successful")
    print(f"📄 Report saved: {batch_report_filename}")
    
    return batch_results

# =============================================================================
# USAGE EXAMPLES
# =============================================================================
if __name__ == "__main__":
    
    # Enhanced single campaign example
    single_campaign = {
        "lead_name": "DeepLearningAI",
        "industry": "Online Learning Platform", 
        "key_decision_maker": "Andrew Ng",
        "position": "CEO",
        "milestone": "new generative AI course launch",
        "objective": "Partnership opportunity discussion"
    }
    
    # Enhanced batch campaign examples
    batch_campaigns = [
        {
            "lead_name": "DeepLearningAI",
            "industry": "EdTech/Online Learning",
            "key_decision_maker": "Andrew Ng", 
            "position": "CEO",
            "milestone": "generative AI course series launch",
            "objective": "Strategic partnership discussion"
        },
        {
            "lead_name": "Anthropic",
            "industry": "AI Safety & Research",
            "key_decision_maker": "Dario Amodei",
            "position": "CEO", 
            "milestone": "Constitutional AI research publication",
            "objective": "Enterprise AI safety consulting"
        },
        {
            "lead_name": "Hugging Face",
            "industry": "AI/ML Platform",
            "key_decision_maker": "Clement Delangue",
            "position": "CEO",
            "milestone": "Series C funding announcement", 
            "objective": "Platform integration opportunity"
        }
    ]
    
    try:
        print("🎯 Advanced Customer Outreach System")
        print("="*50)
        print("Select execution mode:")
        print("1. Single comprehensive campaign")
        print("2. Batch campaign processing")
        print("3. Custom batch with your own targets")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🚀 Running single comprehensive campaign...")
            results = asyncio.run(run_comprehensive_outreach(single_campaign))
            
            if results:
                company_safe = single_campaign['lead_name'].replace(' ', '_').replace('.', '').lower()
                print(f"\n📁 Generated Files:")
                print(f"✅ lead_research_{company_safe}.md")
                print(f"✅ campaign_strategy_{company_safe}.md")
                
        elif choice == "2":
            print(f"\n🚀 Running batch campaign with {len(batch_campaigns)} companies...")
            results = asyncio.run(run_batch_outreach_campaigns(batch_campaigns, "ai_companies_batch"))
            
            if results:
                print(f"\n📁 Generated Files:")
                for company in results.keys():
                    if results[company]['status'] == 'success':
                        company_safe = company.replace(' ', '_').replace('.', '').lower()
                        print(f"✅ lead_research_{company_safe}.md")
                        print(f"✅ campaign_strategy_{company_safe}.md")
                print("✅ ai_companies_batch_report.md")
                
        elif choice == "3":
            print("\n📝 Custom batch mode - modify the batch_campaigns list in the code")
            print("Add your target companies with the required fields:")
            print("- lead_name, industry, key_decision_maker, position, milestone, objective")
            
        else:
            print("❌ Invalid choice. Please run again and select 1, 2, or 3.")
            
    except KeyboardInterrupt:
        print("\n🛑 Campaign interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        print(traceback.format_exc())