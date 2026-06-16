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

def QVA(brain, file_addr, descriptions, cell_knowledges):
    knowledge_addr = "datasets/"+file_addr+"/knowledge/multi_view_cell_knowledge.json"
    updated_knowledges = []
    for i in cell_knowledges:
        check = i.get('check', False)
        number = i.get('number', 0)
        cell_description = descriptions[number]
        cell_tissue = i.get('cell_tissue', '')
        cell_role = i.get('cell_role', '')
        if check == True:
            updated_knowledges.append(i)
            continue

        # Construct system prompt for AI validation
        sys = ("You are a biomedical information validation assistant. "
            "Validate structured cell knowledge extracted from an original cell description. "
            "Check whether each extracted field is supported by the original description and correctly categorized. "
            "Validate these fields: \"cell_tissue\", \"cell_role\". "
            "\"cell_tissue\" means tissue, organ, anatomical region, or anatomical layer. "
            "\"cell_role\" means biological processes, regulatory processes, or core biological functions. "
            "Return only a minified JSON object with exactly this schema: "
            "{"
            "\"ok\": true|false, "
            "\"cell_tissue\": \"\", "
            "\"cell_role\": \"\", "
            "}. "
            "Set ok=true only if all field values are right. "
        )

        # Construct user content with original description and extracted fields
        json_data = {
            'cell_tissue': cell_tissue,
            'cell_role': cell_role
        }
        user_content = (
            f"Original description: {cell_description}\n"
            f"Extracted JSON: {json.dumps(json_data, ensure_ascii=False)}"
        )

        prompt = [{"role": "system", "content": sys},{"role": "user", "content": user_content}]
        answer = brain.chat(prompt)
        data = _safe_json(answer)

        cell_check = data.get("ok", False)
        if cell_check:
            i['check'] = True
            updated_knowledges.append(i)
            continue
        else:
            i['check'] = True
            i["cell_tissue"] = data.get("cell_tissue", "")
            i["cell_role"] = data.get("cell_role", "")
            updated_knowledges.append(i)

    with open(knowledge_addr, 'w', encoding='utf-8') as f:
        json.dump(updated_knowledges, f, ensure_ascii=False, indent=4)

    return updated_knowledges