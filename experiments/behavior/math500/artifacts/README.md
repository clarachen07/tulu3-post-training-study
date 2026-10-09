# Historical artifact inventory

Former release tag: `exp1-behavior-eval-v1`. The original repository and hosted
attachments are retired without additional backups at the owner's request.
There is no supported download for these assets in this publication. The names,
file inventories and hashes below describe historical evidence only.

| Asset | Contents |
|---|---|
| `exp1-shared-chat-raw-jsonl.tar.zst` | All 15 formal shared-chat JSONL files |
| `exp1-native-and-template-diagnostic.tar.zst` | Four native/no-chat seed-1 JSONLs and DPO/RLVR 100-question diagnostic |
| `exp1-complete-logs-and-provenance.tar.zst` | Complete shared-chat logs, status markers, frozen source/config snapshots and inventories |
| `SHA256SUMS` | Hashes of all three compressed assets |

The uncompressed shared-chat tree is indexed in `runs/index.csv`. Raw
generations and complete logs are excluded from this Git snapshot. Full
rescoring is possible only if the caller separately has all exact indexed
files; public aggregate verification requires none of these assets.
