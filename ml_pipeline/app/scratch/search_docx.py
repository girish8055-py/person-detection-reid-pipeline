import docx
import os
import glob

docx_files = glob.glob(r"D:\deployment_model\*.docx") + glob.glob(r"D:\deployment_model\ml_pipeline\app\*.docx")
keywords = ["basic", "auth", "token", "password", "username", "authorization", "bearer", "credentials"]

for file_path in docx_files:
    print(f"\nSearching file: {file_path}")
    try:
        doc = docx.Document(file_path)
        for idx, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if not text:
                continue
            for kw in keywords:
                if kw in text.lower():
                    print(f"  [Para {idx}]: {text}")
                    break
                    
        for t_idx, table in enumerate(doc.tables):
            for r_idx, row in enumerate(table.rows):
                for c_idx, cell in enumerate(row.cells):
                    text = cell.text.strip()
                    if not text:
                        continue
                    for kw in keywords:
                        if kw in text.lower():
                            print(f"  [Table {t_idx}, Row {r_idx}, Col {c_idx}]: {text[:150]}")
                            break
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
