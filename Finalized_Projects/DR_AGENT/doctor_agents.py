#doctor_agents.py
import logging

# --- KEY CHANGE: Import SequentialAgent and LlmAgent ---
from google.adk.agents import LlmAgent, SequentialAgent

from prompts import (DIAGNOSTIC_PROMPT_SEQ, MEDICAL_SPECIALIST_PROMPT_SEQ,
                     TRIAGE_PROMPT_SEQ)
from utils_model import get_available_gemini_models

# ==============================================================================
# --- Initial Setup & Configuration ---
# ==============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dynamically select the best available model for the agents
available_models = get_available_gemini_models()
if not available_models:
    raise RuntimeError("No available Gemini models found. Halting execution.")
SELECTED_MODEL = available_models[0]
logger.info(f"Using selected Gemini model for all sub-agents: {SELECTED_MODEL}")

# ==============================================================================
# --- Agent 1: Triage Agent ---
# ==============================================================================

triage_agent = LlmAgent(
    name="TriageAgent",
    model=SELECTED_MODEL,
    description="First agent in the pipeline. Analyzes patient complaint and all data artifacts (PDFs, images) to extract and structure medical information.",
    instruction=TRIAGE_PROMPT_SEQ,
    # This key stores the agent's output in the session state for the next agent
    output_key="triage_output",
)

# ==============================================================================
# --- Agent 2: Diagnostic Agent ---
# ==============================================================================

diagnostic_agent = LlmAgent(
    name="DiagnosticAgent",
    model=SELECTED_MODEL,
    description="Second agent in the pipeline. Takes structured triage data and provides a neutral, objective analysis of lab and imaging results.",
    instruction=DIAGNOSTIC_PROMPT_SEQ,
    # This key stores the diagnostic report for the final agent
    output_key="diagnostic_output",
)

# ==============================================================================
# --- Agent 3: Medical Specialist Agent ---
# ==============================================================================

specialist_agent = LlmAgent(
    name="MedicalSpecialistAgent",
    model=SELECTED_MODEL,
    description="Final agent in the pipeline. Synthesizes all prior information (triage data and diagnostic report) to generate a final diagnosis and patient-facing report.",
    instruction=MEDICAL_SPECIALIST_PROMPT_SEQ,
    # This is the final output key that the main application will look for
    output_key="workflow_final_output",
)

# ==============================================================================
# --- Root Agent: The Sequential Pipeline ---
# ==============================================================================

# This orchestrator agent automatically runs the sub-agents in the specified order.
# The output of one agent is automatically passed as input to the next.
sequential_medical_agent = SequentialAgent(
    name="MedicalAnalysisPipeline",
    description="A multi-step agent pipeline that triages, diagnoses, and reports on patient data.",
    sub_agents=[triage_agent, diagnostic_agent, specialist_agent],
)
