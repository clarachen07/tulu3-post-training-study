import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


tokenizer_path = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

result = {}
for filename in sys.argv[2:]:
    lengths = []
    with Path(filename).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for text in row["code"]:
                lengths.append(len(tokenizer.encode(text or "", add_special_tokens=False)))
    values = np.asarray(lengths, dtype=np.int64)
    result[filename] = {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
    }

print(json.dumps(result, ensure_ascii=False, indent=2))
