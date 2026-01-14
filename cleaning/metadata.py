import os
import json
import re

# Get the project root directory (parent of cleaning folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Read the cleaned file
input_path = os.path.join(project_root, "data/ration_card/clean/ration_card_kerala_clean.txt")
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Parse sections using regex
# Split by section headers (## SECTION_NAME)
section_pattern = r"## ([A-Z_]+(?:\s*\([A-Z]+\))?)"
sections = re.split(section_pattern, text)

# Build chunks
chunks = []
service = "ration_card"
state = "Kerala"

# sections[0] is empty (before first ##), then alternates: section_name, content, section_name, content...
i = 1
while i < len(sections):
    section_name = sections[i].strip()
    section_text = sections[i + 1].strip() if i + 1 < len(sections) else ""
    
    if section_text:  # Only add non-empty sections
        chunks.append({
            "service": service,
            "state": state,
            "section": section_name,
            "text": section_text
        })
    
    i += 2

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "data/ration_card/chunks")
os.makedirs(output_dir, exist_ok=True)

# Write JSON output
output_path = os.path.join(output_dir, "ration_card_chunks.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"Created {len(chunks)} chunks in {output_path}")
for chunk in chunks:
    print(f"  - {chunk['section']}: {len(chunk['text'])} characters")
