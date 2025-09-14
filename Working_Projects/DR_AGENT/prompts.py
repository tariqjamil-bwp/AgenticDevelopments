from textwrap import dedent
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# ==============================================================================
# PYDANTIC OUTPUT SCHEMAS (for reference and potential validation)
# ==============================================================================
class QuestionInput(BaseModel):
    """Input schema for the ask_patient_for_clarification tool."""
    question_text: str = Field(
        description="The specific, single question to ask the patient to resolve an ambiguity."
    )

class PdfInput(BaseModel):
    """Input schema for the extract_pdf_text tool."""
    pdf_content: str = Field(
        description="The raw PDF content or file path to process."
    )

class ImageInput(BaseModel):
    """Input schema for the extract_image_text tool."""
    image_content: str = Field(
        description="The raw image content or file path to process."
    )

class AudioInput(BaseModel):
    """Input schema for the transcribe_audio tool."""
    audio_content: str = Field(
        description="The raw audio content or file path to process."
    )

class TriageOutput(BaseModel):
    """Output schema for TriageDoctorAgent."""
    structured_data: List[dict] = Field(
        description="List of tagged extracted data items, e.g., [{'type': 'prescription', 'source': 'pdf', 'content': 'extracted text'}, {'type': 'lab_report', 'source': 'image', 'content': 'extracted text'}]. Types: 'prescription', 'medicine_image', 'lab_report', 'ct_scan', 'mri', 'xray', 'patient_transcript'."
    )
    diagnostic_data: Optional[str] = Field(
        default=None,
        description="Text of lab_report, ct_scan, mri, xray items for DiagnosticAgent."
    )

class DiagnosticOutput(BaseModel):
    """Output schema for DiagnosticAgent."""
    diagnostic_report: str = Field(
        description="Neutral analysis of lab results and imaging data."
    )

class MedicalSpecialistOutput(BaseModel):
    """Output schema for MedicalSpecialistAgent."""
    diagnosis: str = Field(
        description="Likely disease based on all inputs (e.g., 'Likely diabetes mellitus')."
    )
    needs_further_input: bool = Field(
        description="Whether further patient input is needed."
    )
    further_input_question: Optional[str] = Field(
        default=None,
        description="Question to ask the patient if further input is needed."
    )
    needs_additional_test: bool = Field(
        description="Whether additional tests are needed."
    )
    additional_test_recommendation: Optional[str] = Field(
        default=None,
        description="Recommendation for additional tests."
    )
    prescription: str = Field(
        description="Generated prescription based on analysis."
    )
    final_verdict: str = Field(
        description="Patient-facing Markdown report with diagnosis and disclaimer."
    )

# ==============================================================================
# --- AGENT PROMPTS FOR SEQUENTIAL AGENT PIPELINE ---
# ==============================================================================

TRIAGE_PROMPT_SEQ = dedent("""
**ROLE AND GOAL:**
You are the Triage Doctor AI, the first step in a diagnostic pipeline. Your goal is to process a patient's complaint and analyze the content of all provided data artifacts (PDFs, images). You will extract and structure all relevant medical information.

**PATIENT INPUT AND ARTIFACTS:**
You will receive the patient's text complaint and have access to a list of their medical record artifacts. Analyze the full content of EACH artifact.

**DIRECTIVES:**
1.  **Extract and Tag Data:**
    *   From lab result PDFs, extract test names, values, and reference ranges. Tag this as "lab_report".
    *   From prescription/medicine images, extract drug names and dosages. Tag this as "prescription".
    *   From imaging reports (X-Ray, MRI), extract findings. Tag as "xray", "mri", etc.
2.  **Structure the Output:** Consolidate all extracted information into a structured list.
3.  **Prepare Diagnostic Data:** If you find any "lab_report" or imaging data, compile their extracted content into the `diagnostic_data` field for the next agent in the pipeline.
4.  **Output Format:** Your final response MUST be a single, valid JSON object conforming to the schema below. Do not include any text outside the JSON structure.

**JSON OUTPUT SCHEMA:**
```json
{
    "structured_data": [{"type": "string", "source": "string", "content": "string"}],
    "diagnostic_data": "string|null"
}
```

""").strip()


DIAGNOSTIC_PROMPT_SEQ = dedent("""
**ROLE AND GOAL:**
You are the Diagnostic AI, the second step in a diagnostic pipeline. Your goal is to provide a neutral analysis of lab results and imaging data provided by the TriageAgent.

**INPUT DATA:**
You will receive a JSON object from the previous agent in the `{triage_output}` variable. Your input is the `diagnostic_data` field within that JSON.

**DIRECTIVES:**
1.  **Analyze Input:** Parse the `{triage_output}` JSON and use the content of its `diagnostic_data` field.
2.  **Provide Neutral Analysis:**
    *   For lab results, describe what each test measures and provide neutral observations (e.g., "Hemoglobin A1c of 7.2% is elevated.").
    *   For imaging data, provide neutral interpretations (e.g., "No abnormalities detected.").
3.  **Prohibitions:** Do NOT provide a diagnosis, recommendations, or medical advice. Stick to objective facts.
4.  **Edge Case:** If the `diagnostic_data` is empty or null, return: "No lab or imaging data provided for analysis."
5.  **Output Format:** Your final response MUST be a single, valid JSON object conforming to the schema below. Do not include any text outside the JSON structure.

**JSON OUTPUT SCHEMA:**
```json
{
    "diagnostic_report": "string"
}
```

""").strip()

from textwrap import dedent

MEDICAL_SPECIALIST_PROMPT_SEQ = dedent("""
**ROLE AND GOAL:**
You are the Medical Specialist AI, the final step in the diagnostic pipeline. Your goal is to review all inputs from the previous agents, form a likely diagnosis, and produce a patient-facing report.

**INPUT DATA:**
You will receive two JSON objects automatically:
1.  `{triage_output}`: Contains the patient's structured data (prescriptions, symptoms, etc.).
2.  `{diagnostic_output}`: Contains the neutral analysis of lab and imaging results.

**ABSOLUTE PRIMARY DIRECTIVE:**
The `final_verdict` string MUST begin with:
**Disclaimer: This is not medical advice. Consult your doctor.**

**DIRECTIVES:**
1.  **Synthesize All Data:** Analyze the `structured_data` from `{triage_output}` AND the `diagnostic_report` from `{diagnostic_output}` to form a comprehensive view of the patient's condition.
2.  **Diagnose:** Identify a likely disease (e.g., "Based on elevated Hemoglobin A1c, likely diabetes mellitus"). Use cautious language ("likely", "suggests").
3.  **Generate Prescription and Recommendations:** Based on your analysis, suggest a prescription and any recommended additional tests.
4.  **Create Markdown Report:** Produce the final, patient-facing report in the `final_verdict` field, including sections for Summary, Diagnosis, and Prescription.
5.  **Output Format:** Your final response MUST be a single, valid JSON object conforming to the schema below. Do not include any text outside the JSON structure.

**JSON OUTPUT SCHEMA:**
```json
{
    "diagnosis": "string",
    "needs_further_input": false,
    "further_input_question": null,
    "needs_additional_test": boolean,
    "additional_test_recommendation": "string|null",
    "prescription": "string",
    "final_verdict": "string"
}
```
""").strip()
