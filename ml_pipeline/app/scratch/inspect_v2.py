import docx

doc = docx.Document("d:\\deployment_model\\Person_Detection_ReID_Pipeline_Documentation_v2.docx")

print(f"Total Paragraphs: {len(doc.paragraphs)}")
print(f"Total Tables: {len(doc.tables)}")

# Print paragraphs containing relevant keywords
keywords = ["lat", "lon", "gps", "telemetry", "json", "result", "schema", "payload", "distance"]
for idx, para in enumerate(doc.paragraphs):
    text = para.text.strip()
    if not text:
        continue
    for kw in keywords:
        if kw in text.lower():
            print(f"[Para {idx}]: {text}")
            break

# Print table cells containing relevant keywords
for t_idx, table in enumerate(doc.tables):
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            text = cell.text.strip()
            for kw in keywords:
                if kw in text.lower():
                    print(f"[Table {t_idx}, Row {r_idx}, Col {c_idx}]: {text[:150]}")
                    break
