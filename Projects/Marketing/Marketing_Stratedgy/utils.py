# Simple Markdown to PDF Converter
# pip install markdown2 weasyprint

import markdown2
from weasyprint import HTML, CSS
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def md_to_pdf(markdown_content, output_path, title="Report"):
    """Convert markdown to PDF - Simple & Effective"""
    
    # Convert markdown to HTML
    html = markdown2.markdown(markdown_content, extras=['tables', 'fenced-code-blocks'])
    
    # Professional CSS styling
    css = """
    @page { size: A4; margin: 2cm; }
    body { font-family: Arial; font-size: 11pt; line-height: 1.6; }
    h1 { color: #2c3e50; font-size: 20pt; }
    h2 { color: #34495e; font-size: 16pt; border-bottom: 2pt solid #3498db; }
    table { width: 100%; border-collapse: collapse; margin: 15pt 0; }
    th { background: #3498db; color: white; padding: 8pt; }
    td { padding: 8pt; border: 1pt solid #ddd; }
    tr:nth-child(even) { background: #f9f9f9; }
    code { background: #f4f4f4; padding: 2pt; }
    pre { background: #2c3e50; color: white; padding: 10pt; }
    """
    
    # Create full HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html><head><title>{title}</title></head>
    <body>{html}</body></html>
    """
    
    # Generate PDF
    HTML(string=full_html).write_pdf(output_path, stylesheets=[CSS(string=css)])
    print(f"✅ PDF created: {output_path}")

# Usage Examples
if __name__ == "__main__":
    
    # Example 1: From string
    sample_md = """
    # Company Report
    ## Executive Summary
    This is a **sample report** with:
    - Key findings
    - Market analysis

    | Metric | Value |
    |--------|-------|
    | Revenue | $1.2B |
    | Growth | +15% |
    """
    
    with open('marketing_report1.md', 'r') as f:
        content = f.read()
    md_to_pdf(content, "marketing_report1.pdf", "Marketing Report")
    
    # Example 2: From file
    # with open('report.md', 'r') as f:
    #     content = f.read()
    # md_to_pdf(content, "output.pdf")
    
    # Example 3: For agent output
    # report = root_agent.run("Analyze Company X")
    # md_content = report.get('comprehensive_company_report')
    # md_to_pdf(md_content, "Company_X_Report.pdf", "Company X Analysis")