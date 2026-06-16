import os
from typing import List
import json

def _safe_json(raw):
    try:
        s = (raw or "").strip()
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j != -1 and j > i:
            import json
            return json.loads(s[i:j+1])
    except Exception:
        pass
    return None

def multi_view_generation(brain, cell_name, cell_description):
    sys = ("You are a biomedical information extraction assistant."
           "Extract the following four fields:"
           "- \"cell_tissue\": tissue, organ, anatomical region, or anatomical layer"
           "- \"cell_role\": biological processes, regulatory processes, or core biological functions"
           "Rules:"
           "1. Output JSON only."
           "2. Each field must be strings with short forms such as: verb + noun or noun phrase."
           "3. Remove redundancy and keep only the most informative phrases.Each item must be concise and normalized."
           "4. If a field is not explicitly supported by the text, return an empty string."
           )
    user_content = (
        f"Cell name: {cell_name}\n"
        f"Description: {cell_description}\n"
    )
    prompt = [{"role": "system", "content": sys}, {"role": "user", "content": user_content}]
    answer = brain.chat(prompt)
    data = _safe_json(answer)
    if not data:
        return "", "", "", ""
    cell_tissue = data.get("cell_tissue", "")
    cell_role = data.get("cell_role", "")
    return cell_tissue, cell_role


def KRA(brain, file_addr: str, label_total: List[str], descriptions: List[str], markers: List[List[str]], config):
    n = len(label_total)
    knowledge_addr = "datasets/" + file_addr + "/knowledge/multi_view_cell_knowledge.json"
    if os.path.exists(knowledge_addr):
        with open(knowledge_addr, 'r', encoding='utf-8') as f:
            cell_knowledges = json.load(f)
    else:
        cell_knowledges = []
        for i in range(n):
            cell_name = label_total[i]
            cell_description = descriptions[i]
            cell_marker = markers[i]
    
            cell_tissue, cell_role = multi_view_generation(brain, cell_name, cell_description)
            temp = {
                "check": False,
                "number": i,
                "type": cell_name,
                "cell_type": cell_name,
                "cell_marker": [str(item) for item in cell_marker[:config.marker_len]],
                "cell_tissue": cell_tissue,
                "cell_role": cell_role,
            }
            cell_knowledges.append(temp)
        with open(knowledge_addr, 'w', encoding='utf-8') as f:
            json.dump(cell_knowledges, f, ensure_ascii=False, indent=4)
    
    return cell_knowledges