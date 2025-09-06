# md_to_pdf_converter.py

import pdfkit
import markdown
import argparse
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
def find_wkhtmltopdf_path():
    """Finds the path to the wkhtmltopdf executable."""
    if sys.platform == "win32":
        # Check the default installation path on Windows
        default_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
        if os.path.exists(default_path):
            return default_path
    else:
        # For macOS and Linux, 'which' command can find the path
        # if it's in the system's PATH.
        from shutil import which
        path = which("wkhtmltopdf")
        if path:
            return path
            
    # If not found in common locations, return None
    return None

def convert_md_to_pdf(input_file: str, output_file: str):
    """
    Converts a Markdown file to a PDF file with basic styling.
    """
    # --- 1. Find the wkhtmltopdf executable ---
    wkhtmltopdf_path = find_wkhtmltopdf_path()
    if not wkhtmltopdf_path:
        print("❌ Error: 'wkhtmltopdf' command-line tool not found.")
        print("Please install it from https://wkhtmltopdf.org/downloads.html and ensure it's in your system's PATH or a standard location.")
        sys.exit(1)
        
    config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

    # --- 2. Check if input file exists ---
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found at '{input_file}'")
        sys.exit(1)

    print(f"📖 Reading markdown from '{input_file}'...")
    with open(input_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    # --- 3. Convert Markdown to HTML ---
    html_content = markdown.markdown(md_content, extensions=['fenced_code', 'tables'])

    # --- 4. Add CSS for styling to make the PDF look professional ---
    # This makes a huge difference in readability.
    html_with_style = f"""
    <html>
      <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #fff;
                margin: 40px;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #222;
                line-height: 1.2;
            }}
            code {{
                background-color: #f6f8fa;
                border-radius: 3px;
                padding: 0.2em 0.4em;
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            }}
            pre {{
                background-color: #f6f8fa;
                padding: 16px;
                overflow: auto;
                border-radius: 3px;
            }}
            pre code {{
                padding: 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 1em;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
      </head>
      <body>
        {html_content}
      </body>
    </html>
    """

    # --- 5. Generate PDF from the HTML string ---
    # We enable local file access to allow images in the markdown to be loaded.
    options = {
        'enable-local-file-access': None,
        'encoding': "UTF-8",
    }
    
    try:
        print(f"⚙️ Generating PDF...")
        pdfkit.from_string(html_with_style, output_file, configuration=config, options=options)
        print(f"✅ Successfully created PDF: '{output_file}'")
    except Exception as e:
        print(f"❌ An error occurred during PDF generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Convert a Markdown file to a PDF.")
    
    #parser.add_argument("input_file", help="The path to the input Markdown (.md) file.")
    
    #parser.add_argument("-o", "--output", help="The path for the output PDF file. If not provided, it will be the same name as the input file with a .pdf extension.")

    #args = parser.parse_args()

    # Determine the output file path
    #output_file_path = args.output
    if True:#not output_file_path:
        # If no output is specified, create it from the input filename
        path = '/home/tjamil/Insync/MyLearning/Dev202xAgents/AgenticDevelopments/Projects/blog_post_writer/final_blog_post_professional.md'
        base_name = os.path.splitext(path)[0]
        output_file_path = f"{base_name}.pdf"
        
    convert_md_to_pdf(path, output_file_path)