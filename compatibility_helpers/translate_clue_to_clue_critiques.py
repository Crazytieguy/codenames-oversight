import json
from sys import stdin

for line in stdin:
    data = json.loads(line)
    if "clues" in data:
        data["clue_critiques"] = [
            {"clue": clue, "critiques": []} for clue in data.pop("clues")
        ]
    elif "oversights" in data:
        for oversight in data["oversights"]:
            oversight["clue_critiques"] = {
                "clue": oversight.pop("clue"),
                "critiques": [],
            }
    else:
        raise ValueError(f"Unknown data format: {data}")
    print(json.dumps(data))
