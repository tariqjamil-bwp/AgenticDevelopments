### `ai_doctor.py` - Corrected Version with Proper ADK Parts
import logging
import asyncio
import os

from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from doctor_agents import sequential_medical_agent
from utils import build_adk_parts_from_folder
from utils_save import save_json_response_to_md

# ==============================================================================
# --- Configuration and Constants ---
# ==============================================================================

APP_NAME = "medical_ai_doctor_app"
USER_ID = "patient_5678"
SESSION_ID = "session_9876"
PATIENT_DATA_FOLDER = "patient_data"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# --- Initial State ---
# ==============================================================================

async def build_initial_state() -> dict:
    """Initialize the session state."""
    logger.info("--- Building initial session state ---")
    return {"initialized": True}

# ==============================================================================
# --- Main Orchestration ---
# ==============================================================================

async def main():
    """Main function to run the medical workflow and generate a report."""
    logger.info("--- Starting medical workflow ---")

    # --- 1. Initialize Services and Session ---
    initial_state = await build_initial_state()
    artifact_service = InMemoryArtifactService()
    session_service = InMemorySessionService()
    
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state
    )

    # Initialize Runner with artifact service
    runner = Runner(
        agent=sequential_medical_agent,
        app_name=APP_NAME,
        session_service=session_service,
        artifact_service=artifact_service
    )

    patient_complaint = input("Please describe your medical complaint: ").strip()
    
    logger.info("--- Kicking off medical analysis pipeline ---")
    
    # Load patient files as ADK Parts (not Artifacts - they go in message parts)
    patient_file_parts = build_adk_parts_from_folder(PATIENT_DATA_FOLDER)
    
    if patient_file_parts:
        logger.info(f"Loaded {len(patient_file_parts)} patient files as ADK Parts")
    else:
        logger.warning("No patient files found - analysis will be based on complaint only")
    
    # Create message parts: text part + file parts
    message_parts = [
        types.Part(text=f"""Please analyze the patient complaint in light of the patient history provided in the attached files.

Patient complaint: {patient_complaint}

Please provide a comprehensive medical analysis including:
1. Possible diagnoses
2. Recommended next steps
3. Any urgent concerns
4. Additional information that would be helpful""")
    ]
    
    # Add file parts to the message
    message_parts.extend(patient_file_parts)
    
    # Create content with text and file parts
    content = types.Content(
        role='user',
        parts=message_parts
    )

    final_response_text = "Agent did not produce a final response."
    
    try:
        async for event in runner.run_async(
            user_id=USER_ID, 
            session_id=SESSION_ID, 
            new_message=content
        ):
            # Log events for debugging (uncomment if needed)
            # logger.debug(f"Event: {event.event_type} from {event.author}")
            
            if event.is_final_response() and event.author == 'MedicalSpecialistAgent':
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif hasattr(event, 'actions') and event.actions and hasattr(event.actions, 'escalate'):
                    final_response_text = f"Agent escalated: {getattr(event, 'error_message', 'No specific message.')}"
                break
                
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        final_response_text = f"Error occurred during analysis: {e}"

    print(f"\n{'='*60}")
    print("MEDICAL AI ANALYSIS RESULT")
    print('='*60)
    print(final_response_text)
    print('='*60)
    print("--- Medical Triage Workflow Complete ---")

    save_json_response_to_md(final_response_text)

if __name__ == "__main__":
    if not os.path.isdir(PATIENT_DATA_FOLDER):
        os.makedirs(PATIENT_DATA_FOLDER)
        print(f"Created '{PATIENT_DATA_FOLDER}' directory.")
        print("Please add patient files (PDFs, images, text files) there before running.")
        exit(1)
    
    # Check if there are any files in the patient data folder
    files = [f for f in os.listdir(PATIENT_DATA_FOLDER) 
             if os.path.isfile(os.path.join(PATIENT_DATA_FOLDER, f))]
    
    if not files:
        print(f"No files found in '{PATIENT_DATA_FOLDER}' directory.")
        print("The analysis will proceed with just the patient complaint.")
        proceed = input("Continue? (y/n): ").strip().lower()
        if proceed != 'y':
            exit(0)
    
    asyncio.run(main())