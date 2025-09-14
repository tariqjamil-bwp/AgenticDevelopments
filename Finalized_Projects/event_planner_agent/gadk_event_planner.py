# -*- coding: utf-8 -*-
"""
Google ADK Event Planning Multi-Agent System
Converted from CrewAI to Google ADK for automated event planning
Updated with Planning Guide Agent instead of Marketing Agent
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
from typing import Optional, List

# -----------------------------
# Constants and API Key Checks
# -----------------------------
APP_NAME = "event_planning_app"
USER_ID = "user_001"
SESSION_ID = "event_planning_session_001"

if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY environment variable not set.")
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# LLM Model
# -----------------------------
AGENT_MODEL = LiteLlm("gemini/gemini-2.5-flash")

# -----------------------------
# Define Asynchronous Tool Functions
# -----------------------------
async def search_internet(query: str) -> str:
    """Search the internet for event planning information."""
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
        results = response_json['organic'][:5]
        string = [f"Title: {r.get('title', 'N/A')}\nLink: {r.get('link', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')}\n---" for r in results]
        return '\n'.join(string)
    except Exception as e:
        return f"An error occurred during search: {e}"

async def scrape_website(website_url: str) -> str:
    """Scrape content from a website URL for event planning details."""
    print(f"--- Using Tool: scrape_website with URL: '{website_url}' ---")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(website_url)
        response.raise_for_status()
        return response.text[:8000]  # Limit content length
    except Exception as e:
        return f"An error occurred during scraping: {e}"

async def get_user_input(prompt: str) -> str:
    """Simulate human input for approval/feedback."""
    print(f"\n{'='*60}")
    print(f"HUMAN INPUT REQUIRED")
    print(f"{'='*60}")
    print(f"{prompt}")
    print("-" * 60)
    
    # In a real implementation, you might want to implement actual user input
    # For now, we'll simulate approval
    user_response = input("Your response (or press Enter to approve): ").strip()
    if not user_response:
        return "Approved - looks good!"
    return user_response

# -----------------------------
# Create Tool Instances
# -----------------------------
search_tool = FunctionTool(func=search_internet)
scrape_tool = FunctionTool(func=scrape_website)
user_input_tool = FunctionTool(func=get_user_input)

# -----------------------------
# Pydantic Models for Structured Outputs
# -----------------------------
class VenueDetails(BaseModel):
    """Structured venue information."""
    name: str = Field(description="Venue name")
    address: str = Field(description="Full venue address")
    capacity: int = Field(description="Maximum capacity")
    booking_status: str = Field(description="Current booking status")
    contact_info: Optional[str] = Field(description="Contact information")
    pricing: Optional[str] = Field(description="Pricing information")
    amenities: Optional[List[str]] = Field(description="Available amenities")

class LogisticsDetails(BaseModel):
    """Structured logistics information."""
    catering: dict = Field(description="Catering arrangements")
    equipment: dict = Field(description="Equipment setup details")
    timeline: dict = Field(description="Event timeline")
    budget_breakdown: dict = Field(description="Cost breakdown")

class PlanningGuideReport(BaseModel):
    """Structured planning guide information."""
    event_type: str = Field(description="Type of event being planned")
    planning_framework: str = Field(description="Planning methodology overview")
    timeline_phases: dict = Field(description="Planning phases and milestones")
    guidelines: dict = Field(description="Detailed planning guidelines")
    risk_management: dict = Field(description="Risk assessment and mitigation")
    quality_standards: List[str] = Field(description="Quality control measures")

# =============================================================================
# AGENT 1: VENUE COORDINATOR
# =============================================================================
venue_coordinator = LlmAgent(
    name="venue_coordinator",
    model=AGENT_MODEL,
    description="Specialized in identifying and booking appropriate venues for events.",
    
    instruction="""
    ## PRIMARY ROLE
    You are a Venue Coordinator with expertise in finding and securing perfect event venues.
    
    ## CORE RESPONSIBILITIES & TASKS
    1. **Venue Search and Selection Task**
       - Find venues in the specified event city that meet criteria
       - Use `search_internet` and `scrape_website` tools extensively
       - Research venue capacity, amenities, pricing, and availability
       - Evaluate venues against event requirements (theme, size, budget)
       - Present detailed venue options with all necessary information
       - Seek human approval for final venue selection using `get_user_input` tool
    
    ## VENUE EVALUATION CRITERIA
    - **Capacity**: Must accommodate expected participants comfortably
    - **Location**: Accessible in specified city with good transportation
    - **Amenities**: Required facilities (AV equipment, catering kitchen, parking)
    - **Budget**: Pricing within specified budget constraints
    - **Availability**: Open on tentative date
    - **Atmosphere**: Matches event theme and professional requirements
    
    ## RESEARCH METHODOLOGY
    1. Search for venues by type and location
    2. Scrape venue websites for detailed information
    3. Compare multiple options against criteria
    4. Present top recommendations with pros/cons
    5. Get human feedback on selection
    6. Confirm booking details and requirements
    
    ## EXPECTED OUTPUT FORMAT (JSON Structure)
    Present your findings as detailed venue information that matches this structure:
    
    ```json
    {
        "name": "Venue Name",
        "address": "Complete street address with city and ZIP",
        "capacity": 100,
        "booking_status": "Available/Pending/Confirmed",
        "contact_info": "Phone and email contact",
        "pricing": "Cost details and package information",
        "amenities": ["WiFi", "AV Equipment", "Catering Kitchen", "Parking"]
    }
    ```
    
    ## HUMAN APPROVAL PROCESS
    Before finalizing venue selection:
    1. Present 2-3 top venue options with complete details
    2. Use `get_user_input` tool to ask: "Please review these venue options and let me know your preference or any concerns:"
    3. Incorporate feedback and make final recommendation
    4. Confirm booking status and next steps
    
    ## QUALITY STANDARDS
    - Research multiple venues (minimum 3-5 options)
    - Provide complete information for each venue
    - Include actual pricing and availability when possible
    - Address all event requirements in recommendations
    - Ensure human approval before finalizing selection
    - Present information in clear, structured format
    """,
    
    tools=[search_tool, scrape_tool, user_input_tool],
    output_key="venue_details"
)

# =============================================================================
# AGENT 2: LOGISTICS MANAGER
# =============================================================================
logistics_manager = LlmAgent(
    name="logistics_manager",
    model=AGENT_MODEL,
    description="Expert in managing all event logistics including catering and equipment.",
    
    instruction="""
    ## PRIMARY ROLE
    You are a Logistics Manager specializing in comprehensive event logistics coordination.
    
    ## CORE RESPONSIBILITIES & TASKS
    1. **Comprehensive Logistics Coordination Task**
       - Coordinate catering for specified number of participants
       - Arrange all necessary equipment for the event
       - Create detailed timeline and setup schedule
       - Manage budget allocation across logistics areas
       - Use `search_internet` and `scrape_tool` for vendor research
       - Seek human approval for logistics plan using `get_user_input` tool
    
    ## LOGISTICS AREAS TO MANAGE
    
    ### Catering Coordination
    - Research catering options suitable for event type and participant count
    - Consider dietary restrictions and preferences
    - Arrange appropriate meal/refreshment service
    - Coordinate delivery and serving logistics
    - Budget for catering within overall event budget
    
    ### Equipment and Setup
    - Audio/visual equipment (microphones, projectors, screens)
    - Seating arrangements and furniture
    - Registration/check-in setup
    - Networking equipment if needed
    - Signage and branding materials
    
    ### Timeline Management
    - Setup schedule with vendor coordination
    - Event day timeline with buffer time
    - Breakdown and cleanup schedule
    - Vendor arrival and departure coordination
    
    ## RESEARCH METHODOLOGY
    1. Search for catering vendors in event city
    2. Research equipment rental companies
    3. Compare pricing and service packages
    4. Create comprehensive logistics plan
    5. Get human approval for arrangements
    6. Confirm all bookings and schedules
    
    ## EXPECTED OUTPUT FORMAT (Structured Plan)
    
    ### Catering Arrangements
    - **Vendor**: [Selected catering company with contact info]
    - **Menu**: [Detailed menu options for participant count]
    - **Service Style**: [Buffet/plated/cocktail style]
    - **Timing**: [Service schedule throughout event]
    - **Cost**: [Detailed pricing breakdown]
    - **Special Requirements**: [Dietary restrictions, equipment needs]
    
    ### Equipment Setup
    - **AV Equipment**: [Complete list with vendor/rental details]
    - **Furniture**: [Seating, tables, registration setup]
    - **Technology**: [WiFi, charging stations, presentation equipment]
    - **Setup Timeline**: [Delivery, setup, testing schedule]
    - **Cost**: [Equipment rental costs and deposits]
    
    ### Event Timeline
    - **Day Before**: [Setup and preparation activities]
    - **Event Morning**: [Final preparations and vendor arrivals]
    - **During Event**: [Service schedule and logistics coordination]
    - **Post Event**: [Breakdown and cleanup schedule]
    
    ### Budget Breakdown
    - **Catering**: [Cost per person and total]
    - **Equipment**: [Rental costs and fees]
    - **Setup/Labor**: [Additional service costs]
    - **Contingency**: [Emergency fund allocation]
    - **Total**: [Complete logistics budget]
    
    ## HUMAN APPROVAL PROCESS
    Use `get_user_input` tool to present logistics plan:
    "Please review this comprehensive logistics plan including catering, equipment, and timeline. Do you approve these arrangements or have any modifications?"
    
    ## QUALITY STANDARDS
    - Coordinate all aspects of event logistics seamlessly
    - Stay within specified budget constraints
    - Provide detailed vendor information and contacts
    - Create realistic timelines with buffer time
    - Ensure all participant needs are addressed
    - Get human approval before confirming arrangements
    """,
    
    tools=[search_tool, scrape_tool, user_input_tool],
    output_key="logistics_details"
)

# =============================================================================
# AGENT 3: EVENT PLANNING GUIDE AGENT
# =============================================================================
planning_guide_agent = LlmAgent(
    name="planning_guide_agent",
    model=AGENT_MODEL,
    description="Expert in creating comprehensive event planning guides with specific guidelines and best practices.",
    
    instruction="""
    ## PRIMARY ROLE
    You are an Event Planning Guide Agent specializing in creating detailed planning guidelines and best practices documentation.
    
    ## CORE RESPONSIBILITIES & TASKS
    1. **Comprehensive Planning Guide Creation Task**
       - Create detailed event planning guide with specific guidelines
       - Research best practices for the event type and size
       - Develop step-by-step planning procedures
       - Include risk management and contingency planning
       - Use `search_internet` and `scrape_website` for industry best practices research
       - Create comprehensive planning guide in markdown format
    
    ## PLANNING GUIDE COMPONENTS
    
    ### Event Planning Framework
    - Timeline-based planning approach with milestone checkpoints
    - Resource allocation and budget management guidelines
    - Vendor selection and management procedures
    - Quality assurance and standards compliance
    - Communication protocols and stakeholder management
    
    ### Operational Guidelines
    - Pre-event planning phases and deliverables
    - Event day execution protocols and procedures
    - Post-event evaluation and follow-up guidelines
    - Documentation requirements and record keeping
    - Performance metrics and success measurement criteria
    
    ### Risk Management Procedures
    - Common event risks identification and assessment
    - Preventive measures and mitigation strategies
    - Emergency response procedures and contingency plans
    - Vendor failure backup plans and alternatives
    - Weather, technical, and logistical risk management
    
    ## RESEARCH METHODOLOGY
    1. Research industry best practices for similar events
    2. Analyze successful event case studies and methodologies
    3. Review professional event planning standards and certifications
    4. Compile proven procedures and guidelines from multiple sources
    5. Create actionable, step-by-step planning procedures
    6. Include specific checklists and quality control measures
    
    ## EXPECTED OUTPUT FORMAT (Comprehensive Planning Guide)
    
    # Event Planning Guide: [Event Type] Planning Guidelines
    
    ## Executive Summary
    **Event Type**: [Event category and description]
    **Target Audience**: [Primary participants and stakeholders]
    **Planning Framework**: [Overview of planning approach and methodology]
    **Key Success Factors**: [Critical elements for event success]
    
    ## Planning Timeline & Milestone Framework
    
    ### Phase 1: Initial Planning (8-12 weeks before event)
    #### Week 8-12 Before Event
    **Milestone**: Event Concept and Requirements Definition
    
    **Guidelines:**
    - Define event objectives, target audience, and success metrics
    - Establish preliminary budget and resource requirements
    - Identify key stakeholders and decision-makers
    - Create project timeline with major milestones
    - Begin venue research and initial vendor outreach
    
    **Deliverables:**
    - Event charter and objectives document
    - Preliminary budget and resource plan
    - Stakeholder communication matrix
    - High-level project timeline
    
    **Quality Checkpoints:**
    - [ ] Event objectives clearly defined and measurable
    - [ ] Budget approved by stakeholders
    - [ ] Key personnel assigned and available
    - [ ] Timeline realistic and achievable
    
    ### Phase 2: Venue and Vendor Selection (6-8 weeks before event)
    #### Week 6-8 Before Event
    **Milestone**: Venue Secured and Primary Vendors Contracted
    
    **Guidelines:**
    - Complete venue evaluation using standardized criteria
    - Negotiate contracts with venue and primary vendors
    - Establish backup options for critical services
    - Create vendor management communication protocols
    - Begin detailed logistics planning based on venue capabilities
    
    **Vendor Selection Criteria:**
    - **Experience**: Minimum 3 similar events in past 2 years
    - **References**: At least 3 verifiable client references
    - **Insurance**: Appropriate liability and professional coverage
    - **Capacity**: Demonstrated ability to handle event scale
    - **Backup Plans**: Clear contingency procedures for service failures
    
    **Deliverables:**
    - Venue contract with detailed specifications
    - Primary vendor contracts and service agreements
    - Vendor contact directory with emergency contacts
    - Detailed venue layout and logistics plan
    
    ### Phase 3: Detailed Planning and Preparation (3-6 weeks before event)
    #### Week 3-6 Before Event
    **Milestone**: All Logistics Confirmed and Event Materials Prepared
    
    **Guidelines:**
    - Finalize all catering, equipment, and service arrangements
    - Complete event materials design and production
    - Establish registration and check-in procedures
    - Create detailed event day timeline and staff assignments
    - Conduct vendor coordination meetings and site visits
    
    **Logistics Coordination Standards:**
    - All vendor arrivals scheduled with 30-minute buffers
    - Equipment testing completed 24 hours before event
    - Catering confirmed with final headcount 48 hours prior
    - Backup plans activated and tested for all critical services
    
    ## Operational Planning Guidelines
    
    ### Budget Management Best Practices
    
    #### Budget Allocation Framework
    **Recommended Budget Distribution:**
    - Venue: 30-40% of total budget
    - Catering: 25-35% of total budget
    - Equipment/Technology: 10-15% of total budget
    - Materials/Supplies: 5-10% of total budget
    - Contingency: 10-15% of total budget
    
    #### Cost Control Procedures
    1. **Triple Quote Requirement**: Obtain minimum 3 quotes for services >$500
    2. **Approval Workflow**: Define spending authorization levels
    3. **Change Management**: Document all budget changes with justification
    4. **Regular Reviews**: Weekly budget reviews during active planning
    5. **Contingency Triggers**: Define conditions for contingency fund use
    
    ### Vendor Management Guidelines
    
    #### Vendor Selection Process
    1. **RFP Development**: Create detailed request for proposal with specifications
    2. **Evaluation Matrix**: Score vendors on price, quality, experience, reliability
    3. **Reference Verification**: Contact minimum 3 recent clients
    4. **Contract Negotiation**: Include cancellation, force majeure, and liability clauses
    5. **Performance Monitoring**: Regular check-ins and milestone reviews
    
    #### Vendor Communication Protocols
    - **Weekly Status Calls**: Scheduled updates during active planning
    - **Escalation Procedures**: Clear contact hierarchy for issue resolution
    - **Documentation Standards**: Written confirmation of all changes
    - **Event Day Coordination**: Designated liaison for each vendor category
    
    ### Quality Assurance Framework
    
    #### Pre-Event Quality Checks
    **72 Hours Before Event:**
    - [ ] All vendor confirmations received and verified
    - [ ] Equipment delivered and tested successfully
    - [ ] Venue setup completed according to specifications
    - [ ] Catering arrangements confirmed with final headcount
    - [ ] Staff briefed on roles, procedures, and emergency contacts
    
    **24 Hours Before Event:**
    - [ ] Final walkthrough completed with venue coordinator
    - [ ] All technology tested and backup systems verified
    - [ ] Registration materials prepared and organized
    - [ ] Emergency procedures reviewed with all staff
    - [ ] Weather and transportation conditions assessed
    
    ## Risk Management and Contingency Planning
    
    ### Common Event Risks and Mitigation Strategies
    
    #### Venue-Related Risks
    **Risk**: Venue double-booking or availability issues
    **Prevention**: 
    - Obtain written confirmation with penalty clauses
    - Maintain backup venue options through planning process
    - Conduct site visit 48 hours before event
    
    **Contingency**: 
    - Pre-negotiated backup venue with 24-hour activation
    - Rapid communication plan for venue changes
    - Alternative setup procedures for different spaces
    
    #### Vendor Service Failures
    **Risk**: Caterer, equipment, or service provider cancellation
    **Prevention**:
    - Vendor backup requirements in all contracts
    - Regular confirmation calls and status updates
    - Insurance requirements for all major vendors
    
    **Contingency**:
    - Pre-qualified backup vendors for all critical services
    - Emergency vendor contact list with 24/7 availability
    - Simplified service options that can be activated quickly
    
    #### Technology and Equipment Failures
    **Risk**: AV equipment, internet, or technical system failures
    **Prevention**:
    - Full equipment testing 24 hours before event
    - Backup equipment for all critical systems
    - Technical support on-site during event
    
    **Contingency**:
    - Simplified presentation formats that don't require technology
    - Mobile hotspot backup for internet connectivity
    - Manual registration procedures if systems fail
    
    ### Emergency Response Procedures
    
    #### Medical Emergency Protocol
    1. Immediate response team designation and training
    2. Emergency services contact information readily available
    3. First aid supplies and trained personnel on-site
    4. Clear evacuation procedures and routes posted
    5. Incident documentation and reporting procedures
    
    #### Weather and Natural Disaster Procedures
    1. Weather monitoring 48 hours before outdoor events
    2. Indoor backup location identified and prepared
    3. Communication plan for event changes or cancellations
    4. Refund and rescheduling policies clearly defined
    5. Vendor force majeure clauses activated as needed
    
    ## Event Day Execution Guidelines
    
    ### Timeline Management Best Practices
    
    #### Setup Phase (Day of Event - Morning)
    **6:00 AM - 9:00 AM**: Vendor arrivals and initial setup
    - Venue access and security procedures activated
    - Equipment delivery and installation begins
    - Catering setup and food safety procedures implemented
    - Registration area preparation and material organization
    
    **Guidelines for Setup Management:**
    - Designated coordinator present for all vendor arrivals
    - Setup completion checklist for each functional area
    - Technology testing completed before any other activities
    - Safety walkthrough conducted after major setup completion
    
    #### Event Execution Phase
    **Registration and Welcome** (30 minutes before start):
    - Registration desk fully operational with backup procedures
    - Welcome materials distributed and signage in place
    - Staff positioned and briefed on crowd management
    - Final sound and technology checks completed
    
    **During Event Operations:**
    - Continuous monitoring of all technical systems
    - Real-time coordination with catering and service staff
    - Attendee experience monitoring and issue resolution
    - Documentation of any issues or deviations from plan
    
    ### Staff Coordination and Communication
    
    #### Staff Assignment Framework
    **Event Director**: Overall coordination and decision-making authority
    **Venue Coordinator**: Liaison with venue staff and facility management
    **Registration Manager**: Check-in processes and attendee services
    **Technical Coordinator**: AV, technology, and equipment management
    **Catering Liaison**: Food service coordination and quality control
    
    #### Communication Protocols
    - Designated communication channel (radios, group chat, etc.)
    - Regular check-in schedule every 30 minutes during setup
    - Issue escalation procedures with clear authority levels
    - Emergency communication plan with all stakeholders
    
    ## Post-Event Evaluation and Follow-up
    
    ### Immediate Post-Event Activities (Within 24 Hours)
    
    #### Event Breakdown and Cleanup
    **Guidelines:**
    - Supervised breakdown of all equipment and materials
    - Final venue walkthrough with facility management
    - Return of rented equipment with condition documentation
    - Secure handling of registration materials and attendee data
    
    #### Initial Performance Assessment
    - Staff debrief meeting to capture immediate observations
    - Vendor performance evaluation and feedback collection
    - Financial reconciliation and expense documentation
    - Issue log completion with resolution notes
    
    ### Comprehensive Event Evaluation (Within 1 Week)
    
    #### Success Metrics Analysis
    **Quantitative Measures:**
    - Attendance vs. registration rates
    - Budget performance vs. planned spending
    - Timeline adherence and milestone achievement
    - Vendor performance ratings and service quality scores
    
    **Qualitative Assessment:**
    - Participant feedback collection and analysis
    - Staff satisfaction and process improvement suggestions
    - Vendor relationship evaluation and future recommendations
    - Overall event experience assessment and lessons learned
    
    #### Documentation and Knowledge Management
    **Required Documentation:**
    - Complete financial reconciliation with vendor payments
    - Event timeline with actual vs. planned comparison
    - Vendor performance evaluations and recommendations
    - Lessons learned document with process improvements
    - Photo/video documentation for future reference
    
    ### Future Planning Improvements
    
    #### Process Optimization
    - Update planning templates based on lessons learned
    - Refine vendor selection criteria and evaluation processes
    - Improve timeline estimates based on actual performance
    - Enhance risk management procedures with new insights
    
    #### Relationship Management
    - Maintain relationships with high-performing vendors
    - Document venue-specific procedures and recommendations
    - Build network of reliable backup service providers
    - Create reference materials for similar future events
    
    ## Planning Templates and Checklists
    
    ### Master Planning Checklist
    
    #### 8-12 Weeks Before Event
    - [ ] Event objectives and success criteria defined
    - [ ] Budget established and approved by stakeholders
    - [ ] Project team assembled with defined roles
    - [ ] Initial venue research and site visits scheduled
    - [ ] Save-the-date communications sent to key participants
    
    #### 6-8 Weeks Before Event
    - [ ] Venue contract signed with detailed specifications
    - [ ] Primary vendors selected and contracts executed
    - [ ] Backup options identified for all critical services
    - [ ] Event registration system setup and tested
    - [ ] Initial marketing and communication plan activated
    
    #### 4-6 Weeks Before Event
    - [ ] All logistics confirmed with vendor coordination meeting
    - [ ] Event materials designed, produced, and delivered
    - [ ] Staff assignments finalized with role-specific training
    - [ ] Technology requirements tested at venue
    - [ ] Emergency procedures documented and communicated
    
    #### 2-4 Weeks Before Event
    - [ ] Final headcount confirmed with catering adjustments
    - [ ] Event day timeline distributed to all stakeholders
    - [ ] Venue layout finalized with setup specifications
    - [ ] Registration process tested with sample participants
    - [ ] Weather contingency plans activated if applicable
    
    #### 1 Week Before Event
    - [ ] All vendor confirmations received and documented
    - [ ] Event day staff briefing completed
    - [ ] Emergency contact information verified and distributed
    - [ ] Final venue walkthrough with facility management
    - [ ] Backup plans tested and ready for activation
    
    ### Vendor Evaluation Template
    
    #### Scoring Criteria (1-5 scale, 5 being excellent)
    **Experience and Expertise**: ___/5
    - Years in business and similar event experience
    - Staff qualifications and industry certifications
    - Portfolio of comparable events and client references
    
    **Service Quality and Reliability**: ___/5
    - Responsiveness to inquiries and communications
    - Attention to detail in proposals and planning
    - Reputation for on-time delivery and service execution
    
    **Financial and Contract Terms**: ___/5
    - Competitive pricing for scope of services
    - Transparent pricing with clear inclusions/exclusions
    - Reasonable contract terms and cancellation policies
    
    **Risk Management and Backup Plans**: ___/5
    - Insurance coverage and liability protection
    - Contingency procedures for service failures
    - Emergency contact availability and response procedures
    
    **Total Score**: ___/20
    
    ### Budget Tracking Template
    
    #### Budget Category Tracking
    | Category | Budgeted Amount | Actual Quotes | Contracted Amount | Variance | Notes |
    |----------|----------------|---------------|-------------------|----------|-------|
    | Venue | $____ | $____ | $____ | $____ | |
    | Catering | $____ | $____ | $____ | $____ | |
    | Equipment | $____ | $____ | $____ | $____ | |
    | Materials | $____ | $____ | $____ | $____ | |
    | Staff/Labor | $____ | $____ | $____ | $____ | |
    | Contingency | $____ | $____ | $____ | $____ | |
    | **TOTAL** | **$____** | **$____** | **$____** | **$____** | |
    
    ## Best Practices Summary
    
    ### Key Success Factors
    1. **Early Planning**: Begin detailed planning 8-12 weeks before event
    2. **Stakeholder Engagement**: Regular communication with all parties
    3. **Risk Mitigation**: Identify and plan for potential issues early
    4. **Quality Control**: Multiple checkpoints and verification procedures
    5. **Documentation**: Comprehensive record-keeping throughout process
    
    ### Common Planning Mistakes to Avoid
    1. **Underestimating Setup Time**: Always add buffer time for preparation
    2. **Single-Source Dependencies**: Maintain backup options for critical services
    3. **Poor Communication**: Establish clear protocols and regular check-ins
    4. **Budget Overruns**: Monitor spending closely with approval procedures
    5. **Inadequate Contingency Planning**: Prepare for multiple failure scenarios
    
    ### Planning Excellence Standards
    - Every major decision documented with rationale and alternatives considered
    - All vendor relationships managed professionally with clear expectations
    - Continuous improvement mindset with lessons learned captured and applied
    - Participant experience prioritized in all planning decisions
    - Financial stewardship with transparent budget management and reporting
    
    ## QUALITY STANDARDS
    - Create comprehensive, actionable planning guidelines
    - Include specific procedures, checklists, and templates
    - Focus on risk management and contingency planning
    - Provide realistic timelines with built-in buffer periods
    - Emphasize documentation and continuous improvement
    - Include industry best practices and proven methodologies
    """,
    
    tools=[search_tool, scrape_tool],
    output_key="planning_guide"
)

# =============================================================================
# SEQUENTIAL WORKFLOW WITH PARALLEL SIMULATION
# =============================================================================
event_planning_agent = SequentialAgent(
    name='EventPlanningAgent',
    description="""
    Comprehensive event planning workflow:
    1. Venue Coordinator - Find and secure appropriate venue
    2. Parallel Coordinator - Set up logistics and planning guide coordination
    3. Logistics Manager - Handle catering, equipment, and logistics
    4. Planning Guide Agent - Create comprehensive planning guidelines and best practices
    
    This system automates event planning with human approval checkpoints
    for critical decisions like venue selection and logistics arrangements,
    plus provides detailed planning guidelines for execution.
    """,
    sub_agents=[
        venue_coordinator,
        logistics_manager,
        planning_guide_agent
    ],
)

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================
# Update the file saving logic in the main execution function
async def run_event_planning_workflow(event_details: dict):
    """Run the complete event planning workflow."""
    
    session_service = InMemorySessionService()
    print("🎉 Starting Event Planning Workflow...")
    
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=event_planning_agent, app_name=APP_NAME, session_service=session_service)
    
    # Format the input query
    query = f"""
    Event Planning Request:
    
    Event Topic: {event_details['event_topic']}
    Description: {event_details['event_description']}
    City: {event_details['event_city']}
    Date: {event_details['tentative_date']}
    Expected Participants: {event_details['expected_participants']}
    Budget: ${event_details['budget']}
    Venue Type: {event_details['venue_type']}
    
    Please coordinate complete event planning including venue selection,
    logistics management, and comprehensive planning guide with guidelines.
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
                if agent_name == "venue_coordinator":
                    # Try to extract JSON from venue details
                    try:
                        import re
                        json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
                        if json_match:
                            venue_json = json.loads(json_match.group())
                        else:
                            venue_json = {
                                "name": "Extracted from response",
                                "address": "See full response",
                                "capacity": event_details['expected_participants'],
                                "booking_status": "Researched"
                            }
                        
                        with open("venue_details.json", "w", encoding='utf-8') as f:
                            json.dump(venue_json, f, indent=2)
                        print("✅ Venue details saved to venue_details.json")
                    except Exception as e:
                        print(f"⚠️ Could not save JSON format: {e}")
                
                elif agent_name == "logistics_manager":
                    with open("logistics_plan.md", "w", encoding='utf-8') as f:
                        f.write(content_text)
                    print("✅ Logistics plan saved to logistics_plan.md")
                
                elif agent_name == "planning_guide_agent":
                    with open("event_planning_guide.md", "w", encoding='utf-8') as f:
                        f.write(content_text)
                    print("✅ Planning guide saved to event_planning_guide.md")
                    break  # Final agent completed
    
    except Exception as e:
        print(f"❌ Error during workflow execution: {e}")
        return None
    
    print("\n🎉 Event Planning Workflow Completed Successfully!")
    return final_results

# Update the final file listing section
if __name__ == "__main__":
    # Example event details (matching original CrewAI version)
    event_details = {
        'event_topic': "Daughter's engagement function",
        'event_description': "A gathering of 70 family guests to celebrate engagement function.",
        'event_city': "Karachi",
        'tentative_date': "2025-09-15",
        'expected_participants': 70,
        'budget': 2000,
        'venue_type': "Small Banquet Hall, near safoora"
    }
    
    try:
        results = asyncio.run(run_event_planning_workflow(event_details))
        
        if results:
            print("\n📁 Generated Files:")
            print("- venue_details.json")
            print("- logistics_plan.md")
            print("- event_planning_guide.md")  # Updated file name
            
            # Display venue details if available
            try:
                with open('venue_details.json', 'r') as f:
                    venue_data = json.load(f)
                print("\n📍 Venue Details:")
                print(json.dumps(venue_data, indent=2))
            except FileNotFoundError:
                print("⚠️ venue_details.json not found")
            
    except KeyboardInterrupt:
        print("\n❌ Workflow interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
