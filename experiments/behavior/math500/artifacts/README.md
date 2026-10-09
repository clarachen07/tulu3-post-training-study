# GitHub Release artifacts

Release tag: `exp1-behavior-eval-v1`

Release page:
[`exp1-behavior-eval-v1`](https://github.com/clarachen07/tulu3-behavior-eval/releases/tag/exp1-behavior-eval-v1)

| Asset | Contents |
|---|---|
| `exp1-shared-chat-raw-jsonl.tar.zst` | All 15 formal shared-chat JSONL files |
| `exp1-native-and-template-diagnostic.tar.zst` | Four native/no-chat seed-1 JSONLs and DPO/RLVR 100-question diagnostic |
| `exp1-complete-logs-and-provenance.tar.zst` | Complete shared-chat logs, status markers, frozen source/config snapshots and inventories |
| `SHA256SUMS` | Hashes of all three compressed assets |

The uncompressed shared-chat tree is indexed in `runs/index.csv`. Raw
generations are intentionally excluded from Git history so that future
corrections do not permanently bloat the repository.
