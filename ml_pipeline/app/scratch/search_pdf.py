import os

try:
    import pypdf
    print("pypdf available")
except ImportError:
    try:
        import PyPDF2 as pypdf
        print("PyPDF2 available")
    except ImportError:
        pypdf = None
        print("No pdf reader library available directly")

if pypdf:
    pdf_path = r"D:\deployment_model\ml_pipeline\app\API_Integration_Documentation_v1.pdf"
    if not os.path.exists(pdf_path):
        pdf_path = r"D:\deployment_model\API_Integration_Documentation_v1.pdf"
    
    if os.path.exists(pdf_path):
        print(f"Reading {pdf_path}...")
        reader = pypdf.PdfReader(pdf_path)
        print("Pages:", len(reader.pages))
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            lines = text.split("\n")
            for line in lines:
                if any(w in line.lower() for w in ["basic", "auth", "token", "password", "username", "credential"]):
                    print(f"Page {idx+1}: {line}")
    else:
        print("PDF not found")
else:
    # Try docx if pdf not readable, but we already read docx.
    print("No PDF library. Let's try to extract text from the pdf if possible by running a python command.")
