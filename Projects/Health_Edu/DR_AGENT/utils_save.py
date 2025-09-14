import json
import re
import os
from datetime import datetime
from typing import Any, Dict

def save_json_response_to_md(response_text: str, output_dir: str = "reports", patient_id: str = None) -> str:
    """
    Convert JSON response to markdown file with keys as subheadings.
    
    Args:
        response_text: The complete response text containing JSON
        output_dir: Directory to save the markdown file
        patient_id: Optional patient ID for filename
        
    Returns:
        Path to saved markdown file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and parse JSON
    json_data = extract_json_from_response(response_text)
    if not json_data:
        print("❌ No valid JSON found in response")
        return None
    
    # Convert to markdown
    markdown_content = json_to_markdown(json_data)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"medical_report_{patient_id}_{timestamp}.md" if patient_id else f"medical_report_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    # Save file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"📄 Report saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return None

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract JSON from response text."""
    # Look for JSON in code blocks
    json_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_pattern, response_text, re.DOTALL)
    
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            return {}
    
    # Try parsing entire response as JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {}

def json_to_markdown(data: Dict[str, Any]) -> str:
    """
    Convert JSON dictionary to markdown with keys as subheadings.
    
    Args:
        data: JSON data as dictionary
        
    Returns:
        Formatted markdown string
    """
    lines = []
    
    # Header
    lines.append("# Medical Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Process each key-value pair
    for key, value in data.items():
        if value is None:
            continue
            
        # Create heading from key
        heading = format_key_as_heading(key)
        lines.append(f"## {heading}")
        lines.append("")
        
        # Format value
        formatted_value = format_value(value)
        lines.append(formatted_value)
        lines.append("")
    
    return "\n".join(lines)

def format_key_as_heading(key: str) -> str:
    """
    Convert JSON key to a nice heading.
    
    Args:
        key: JSON key string
        
    Returns:
        Formatted heading string
    """
    # Replace underscores with spaces
    heading = key.replace('_', ' ')
    
    # Convert to title case
    heading = heading.title()
    
    # Handle common medical terms
    replacements = {
        'Needs Further Input': '🤔 Additional Information Needed',
        'Further Input Question': '❓ Follow-up Question',
        'Needs Additional Test': '🧪 Additional Testing Required',
        'Additional Test Recommendation': '📋 Recommended Tests',
        'Prescription': '💊 Prescription & Treatment',
        'Final Verdict': '📄 Detailed Analysis',
        'Diagnosis': '🩺 Diagnosis',
    }
    
    for old, new in replacements.items():
        if heading == old:
            return new
    
    return heading

def format_value(value: Any) -> str:
    """
    Format JSON value for markdown display.
    
    Args:
        value: The value to format
        
    Returns:
        Formatted string
    """
    if isinstance(value, bool):
        return "✅ Yes" if value else "❌ No"
    
    elif isinstance(value, str):
        # Handle multi-line strings with markdown formatting
        if '\n' in value:
            # Split into paragraphs and format
            paragraphs = value.split('\n\n')
            formatted_paragraphs = []
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                # Handle markdown headers in the text
                if para.startswith('## '):
                    formatted_paragraphs.append(f"### {para[3:]}")
                elif para.startswith('**') and para.endswith(':**'):
                    formatted_paragraphs.append(f"**{para}**")
                elif para.startswith('Disclaimer:'):
                    formatted_paragraphs.append(f"> ⚠️ **{para}**")
                else:
                    # Handle bullet points
                    if '*   ' in para:
                        lines = para.split('\n')
                        for line in lines:
                            if line.startswith('*   '):
                                formatted_paragraphs.append(f"- {line[4:]}")
                            else:
                                formatted_paragraphs.append(line)
                    else:
                        formatted_paragraphs.append(para)
            
            return '\n\n'.join(formatted_paragraphs)
        else:
            return value
    
    elif isinstance(value, list):
        if not value:
            return "*None specified*"
        
        formatted_items = []
        for item in value:
            if isinstance(item, str):
                formatted_items.append(f"- {item}")
            else:
                formatted_items.append(f"- {str(item)}")
        return '\n'.join(formatted_items)
    
    elif isinstance(value, dict):
        formatted_items = []
        for k, v in value.items():
            formatted_items.append(f"- **{format_key_as_heading(k)}:** {format_value(v)}")
        return '\n'.join(formatted_items)
    
    else:
        return str(value) if value is not None else "*Not specified*"

# Example usage
def test_conversion():
    """Test the conversion with sample data"""
    sample_response = '''
MEDICAL AI ANALYSIS RESULT
============================================================
```json
{
    "diagnosis": "Lower left abdominal pain (likely functional gastrointestinal, e.g., Irritable Bowel Syndrome); Non-alcoholic Fatty Liver Disease (NAFLD); Dyslipidemia; Mild Impaired Renal Function/Early Chronic Kidney Disease.",
    "needs_further_input": false,
    "further_input_question": null,
    "needs_additional_test": true,
    "additional_test_recommendation": "Further evaluation of kidney function including estimated Glomerular Filtration Rate (eGFR) and urine analysis for protein/albumin. If abdominal pain persists or changes, further gastrointestinal workup may be considered.",
    "prescription": "Continue current medications (Cap NEXIUM, Tab SPATFON, 2EGAP). Implement significant lifestyle changes including a heart-healthy diet, regular physical activity, and weight management.",
    "final_verdict": "Disclaimer: This is not medical advice. Consult your doctor.\\n\\n## Patient Report\\n\\n**Summary of Findings:**\\nYou presented with pain in the lower left abdomen. Tests reveal fatty liver disease, high cholesterol, and elevated creatinine suggesting reduced kidney function."
}
```
============================================================
    '''
    
    return save_json_response_to_md(sample_response, patient_id="TEST123")

if __name__ == "__main__":
    test_conversion()