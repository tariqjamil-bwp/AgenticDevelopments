# -*- coding: utf-8 -*-
"""
AI Journalist Agent using Google ADK with Newspaper3k and DuckDuckGo Search
Restored original prompts, using DuckDuckGo as search tool with query wrapper
"""

import os
import asyncio
import logging
from textwrap import dedent
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from newspaper import Article
from datetime import datetime
from ddgs import DDGS
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
# Suppress OpenTelemetry warnings
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

# Constants
APP_NAME = "ai_journalist_app"
USER_ID = "journalist_user"
SESSION_ID = "journalist_session"
CURRENT_DATE = datetime.now().strftime("%B %d, %Y")

# Verify API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not set")

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# LLM Model
AGENT_MODEL = LiteLlm("gemini/gemini-2.0-flash")

# Python wrapper for DuckDuckGo search
def ddg_news_search(query: str) -> str:
    """Wrapper to search news using DuckDuckGo"""
    print(f"🔍 DDG query: '{query}'")
    if not query or query.strip() == "":
        print("❌ DDG: Empty query")
        return "ERROR: No valid search query provided."
    try:
        with DDGS() as ddgs:
            results = ddgs.news(query, max_results=10)
        if not results:
            print("⚠️ DDG: No results")
            return "No news results found."
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['url']} - {result['source']} - {result['title']}")
        result_str = "\n".join(output)
        print(f"✅ DDG success: {len(result_str)} chars")
        return result_str
    except Exception as e:
        print(f"❌ DDG error: {str(e)}")
        return f"Search failed: {str(e)}"

# Initialize tools
search_tool = FunctionTool(
    func=ddg_news_search
)

async def extract_article_content(url: str) -> str:
    """Extracts article content from a URL using newspaper3k."""
    print(f"🔍 Extracting: {url}")
    try:
        article = Article(url, fetch_images=False, memoize_articles=False)
        article.download()
        article.parse()
        if article.text and len(article.text.strip()) > 100:
            print(f"✅ Extracted {len(article.text)} chars")
            return article.text[:2000]
        print(f"⚠️ No content from {url}")
        return "No content extracted from the article."
    except Exception as e:
        print(f"❌ Extraction failed: {str(e)}")
        return f"Error extracting article from {url}: {str(e)}"

newspaper_tool = FunctionTool(
    func=extract_article_content
)

# =============================================================================
# SEARCHER AGENT
# =============================================================================
searcher = LlmAgent(
    name="Searcher",
    model=AGENT_MODEL,
    description=dedent(
        """\
        You are a world-class journalist for the New York Times. Given a topic, generate a list of 3 search terms
        for writing an article on that topic. Then search the web for each term, analyse the results
        and return the 10 most relevant URLs.
        """
    ),
    instruction=dedent(
        f"""\
        INSTRUCTIONS:
        1. Given a topic, first generate a list of 3 search terms related to the topic.
        2. For each search term, use `InternetNewsSearch(query="exact_search_term")` and analyze the results.
        3. From the results of all searches, return the 10 most relevant URLs to the topic.
        4. Remember: you are writing for the New York Times, so the quality of the sources is important.
        5. Prioritize reputable news sources (NYT, WSJ, BBC, Reuters, AP), academic papers, and authoritative websites.
        6. Include the current date in your analysis for recency: {CURRENT_DATE}.
        
        OUTPUT FORMAT:
        - **Search Terms**: List the 3 search terms you used
        - **Top 10 URLs**: Numbered list with brief description of each source's relevance
        - **Source Quality Assessment**: Brief note on why these sources are NYT-worthy
        - **Publication Dates**: Include publication dates when available
        - **Debug Log**: [Log any tool call issues]
        """
    ),
    tools=[search_tool],
    output_key="search_results"
)

# =============================================================================
# WRITER AGENT
# =============================================================================
writer = LlmAgent(
    name="Writer",
    model=AGENT_MODEL,
    description=dedent(
        """\
        You are a senior writer for the New York Times. Given a topic and a list of URLs,
        your goal is to write a high-quality NYT-worthy article on the topic.
        """
    ),
    instruction=dedent(
        f"""\
        INSTRUCTIONS:
        1. Given a topic and a list of URLs, first use `NewspaperArticleExtractor(url="exact_url")` to read the content from each URL.
        2. Analyze the content from each source for key facts, quotes, and context.
        3. Then write a high-quality NYT-worthy article on the topic.
        4. The article should be well-structured, informative, and engaging.
        5. Ensure the length is at least as long as a NYT cover story -- at a minimum, 15 paragraphs.
        6. Ensure you provide a nuanced and balanced opinion, quoting facts where possible.
        7. Remember: you are writing for the New York Times, so the quality of the article is important.
        8. Focus on clarity, coherence, and overall quality.
        9. Never make up facts or plagiarize. Always provide proper attribution.
        10. Structure the article with: compelling headline, lead paragraph, body with multiple sections, 
            analysis, quotes, and conclusion.
        11. Include the current date ({CURRENT_DATE}) and context in the article.
        
        ARTICLE STRUCTURE:
        - **Headline**: Compelling, informative title (1-2 lines)
        - **Byline**: "By AI Journalist, The New York Times"
        - **Dateline**: {CURRENT_DATE}, New York
        - **Lead Paragraph**: Hook the reader with the most important information (2-3 sentences)
        - **Body**: 12-15 paragraphs with background, analysis, quotes, and context
        - **Conclusion**: Wrap up with implications or next steps
        - **Sources**: List full URLs and publication details with inline citations
        
        CITATION STYLE:
        - Use inline citations: [Source Name, Date]
        - Include a "Sources" section at the end with full URLs and publication details
        - Attribute quotes and facts properly throughout the article
        
        WRITING TIPS:
        - Use active voice and varied sentence structure
        - Include multiple perspectives when relevant
        - Maintain objective journalistic tone
        - Focus on human impact and broader implications
        - Use transitions between sections for smooth flow
        """
    ),
    tools=[newspaper_tool],
    output_key="article_draft"
)

# =============================================================================
# EDITOR AGENT
# =============================================================================
editor = LlmAgent(
    name="Editor",
    model=AGENT_MODEL,
    description="You are a senior NYT editor. Given a topic, your goal is to write a NYT worthy article.",
    instruction=dedent(
        f"""\
        INSTRUCTIONS:
        1. Review the search results from the Searcher agent for source quality and relevance.
        2. Read the draft article from the Writer agent carefully.
        3. Edit, proofread, and refine the article to ensure it meets the high standards of the New York Times.
        4. The article should be extremely articulate and well written.
        5. Focus on clarity, coherence, and overall quality.
        6. Ensure the article is engaging and informative.
        7. Check for factual accuracy, proper attribution, and balanced reporting.
        8. Improve flow, eliminate redundancy, and enhance readability.
        9. Verify the article structure follows NYT standards (headline, byline, lead, body, conclusion).
        10. Remember: you are the final gatekeeper before the article is published.
        11. Include the current date: {CURRENT_DATE}.
        
        EDITORIAL CHECKLIST:
        - [ ] Compelling, accurate headline that draws readers in
        - [ ] Strong lead paragraph that hooks the reader immediately
        - [ ] Well-structured body with clear sections and logical progression
        - [ ] Proper attribution and no plagiarism throughout
        - [ ] Balanced reporting with multiple perspectives where appropriate
        - [ ] Grammar, style, and NYT tone are professional and engaging
        - [ ] Length appropriate for topic depth (minimum 15 paragraphs, 2000+ words)
        - [ ] Sources section complete, accurate, and properly formatted
        - [ ] No factual errors or inconsistencies
        - [ ] Smooth transitions between sections and ideas
        
        SPECIFIC EDITING TASKS:
        1. **Content**: Verify all facts against sources, ensure balance
        2. **Structure**: Improve flow between paragraphs and sections
        3. **Language**: Enhance clarity, eliminate jargon, improve readability
        4. **Style**: Ensure NYT journalistic standards (objective, informative)
        5. **Citations**: Verify all attributions are accurate and complete
        6. **Engagement**: Add compelling details, human elements, context
        
        OUTPUT:
        - **Final Article**: Polished article ready for publication in markdown format
        - **Editor's Notes**: Summary of major changes made and rationale
        - **Quality Score**: 1-10 assessment of the final article with justification
        """
    ),
    output_key="final_article"
)

# =============================================================================
# SEQUENTIAL JOURNALIST WORKFLOW
# =============================================================================
ai_journalist = SequentialAgent(
    name="AIJournalist",
    description="Complete AI Journalist workflow: Search → Write → Edit using DuckDuckGo and Newspaper3k",
    sub_agents=[searcher, writer, editor]
)

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================
async def generate_article(topic: str):
    """Generate a complete NYT-worthy article using the AI Journalist workflow."""
    # Format instructions with topic
    searcher.instruction = searcher.instruction.format(topic=topic)
    writer.instruction = writer.instruction.format(topic=topic)
    editor.instruction = editor.instruction.format(topic=topic)
    
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=ai_journalist, app_name=APP_NAME, session_service=session_service)
    
    query = f"""
    TOPIC: {topic}
    
    Complete NYT Article Generation Workflow ({CURRENT_DATE}):
    
    **SEARCHER**: Generate 3 high-quality search terms and find the 10 most relevant, reputable URLs for this topic.
    Focus on NYT-worthy sources: major news outlets (NYT, WSJ, BBC, Reuters, AP), academic papers, government reports, and authoritative websites.
    Prioritize recent, credible sources suitable for New York Times standards.
    
    **WRITER**: Using the provided URLs, use `NewspaperArticleExtractor` to extract key information from each source, then write a comprehensive 
    NYT cover story (minimum 15 paragraphs, 2000+ words) on the topic. Include proper attribution, balanced 
    reporting, multiple perspectives, and engaging narrative structure. Use inline citations throughout.
    
    **EDITOR**: Review and polish the draft article thoroughly. Ensure it meets NYT journalistic standards 
    for clarity, accuracy, engagement, factual integrity, and professional quality. Provide the final publishable version
    with editor's notes on improvements made.
    
    Goal: Create a publication-ready New York Times article that could appear on the front page.
    
    IMPORTANT: Maintain strict journalistic integrity - verify facts, attribute sources, avoid speculation, 
    provide balanced perspectives, and write in professional NYT style.
    """
    
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    article_stages = {}
    try:
        print(f"🚀 Starting article generation for topic: {topic}")
        print(f"🔍 Testing DuckDuckGo API...")
        test_result = ddg_news_search(query="test query")
        print(f"✅ API test: {len(test_result)} chars")
        
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            if event.is_final_response():
                agent_name = event.author
                content_text = event.content.parts[0].text
                article_stages[agent_name] = content_text
                
                print(f"✅ {agent_name} completed")
                print(f"📝 Output length: {len(content_text)} characters")
                
                safe_topic = topic.replace(' ', '_').replace('/', '_').lower()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if agent_name == "Searcher":
                    filename = f"search_results_{safe_topic}_{timestamp}.txt"
                    with open(filename, "w", encoding='utf-8') as f:
                        f.write(f"# Search Results for: {topic}\n\n")
                        f.write(f"Date: {CURRENT_DATE}\n\n")
                        f.write(content_text)
                    print(f"💾 Saved: {filename}")
                    
                elif agent_name == "Writer":
                    filename = f"article_draft_{safe_topic}_{timestamp}.md"
                    with open(filename, "w", encoding='utf-8') as f:
                        f.write(f"# Article Draft: {topic}\n\n")
                        f.write(f"**Date**: {CURRENT_DATE}\n\n")
                        f.write(content_text)
                    print(f"💾 Saved: {filename}")
                    
                elif agent_name == "Editor":
                    filename = f"final_article_{safe_topic}_{timestamp}.md"
                    with open(filename, "w", encoding='utf-8') as f:
                        f.write(f"# Final NYT Article: {topic}\n\n")
                        f.write(f"**Publication Date**: {CURRENT_DATE}\n\n")
                        f.write(content_text)
                    print(f"💾 Saved: {filename}")
                    break
        
        return article_stages.get("Editor", article_stages.get("Writer", "Failed"))
        
    except Exception as e:
        print(f"❌ Error during article generation: {e}")
        import traceback
        print(traceback.format_exc())
        return None

# =============================================================================
# COMMAND-LINE INTERFACE FOR DEBUGGING
# =============================================================================
if __name__ == "__main__":
    print("AI Journalist Agent 🗞️")
    print(f"Date: {CURRENT_DATE}")
    print("Generate NYT-worthy articles using Gemini + Newspaper3k + DuckDuckGo")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Missing GOOGLE_API_KEY")
        exit(1)
    
    print("✅ API Key ready")
    print("\nHow it works:")
    print("1. Searcher finds 10 sources using DuckDuckGo")
    print("2. Writer extracts content with Newspaper3k, drafts article")
    print("3. Editor polishes to NYT standards")
    print("\nOutput: NYT-style article (15+ paragraphs)")
    
    topic = input("\nTopic (e.g., 'The impact of AI on modern journalism'): ").strip()
    if not topic:
        print("❌ No topic")
        exit(1)
    
    print(f"\n📝 Generating: {topic}")
    print("This may take 2-5 minutes...")
    
    response = asyncio.run(generate_article(topic))
    
    if response:
        print("\n" + "="*60)
        print("🎉 ARTICLE")
        print("="*60)
        print(response[:1500] + "..." if len(response) > 1500 else response)
        print(f"\n📁 Files: {topic.lower().replace(' ', '_')}*")
    else:
        print("\n❌ Failed")
        print("Check logs or test DuckDuckGo: python -c \"from duckduckgo_search import DDGS; print(DDGS().news('test query', max_results=10))\"")