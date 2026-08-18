import docx

doc = docx.Document("d:\\deployment_model\\Person_Detection_ReID_Pipeline_Documentation_v2.docx")

for idx, t in enumerate(doc.tables):
    print(f"--- Table {idx} ---")
    for r_idx, row in enumerate(t.rows):
        for c_idx, cell in enumerate(row.cells):
            print(f"Row {r_idx}, Col {c_idx}: {cell.text.strip()[:100]}")
