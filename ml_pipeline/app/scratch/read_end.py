import docx

doc = docx.Document("d:\\deployment_model\\Person_Detection_ReID_Pipeline_Documentation_v2.docx")

print("--- Paragraphs ---")
for idx, p in enumerate(doc.paragraphs):
    if idx >= 34:
        print(f"[Para {idx}]: {p.text}")

print("--- Table 1 ---")
table = doc.tables[1]
for r_idx, row in enumerate(table.rows):
    for c_idx, cell in enumerate(row.cells):
        print(f"Row {r_idx}, Col {c_idx}:")
        print(cell.text)
