> Historical protocol notes. Complete inputs/outputs and machine-specific synchronization tools are local-only. The canonical public report is `reports/report_zh.md`; full verification requires external evidence.

# Tülu 3 四阶段直接答案复测

固定 1,024 训练题、1,319 测试题，4 个原始 Tülu 3 阶段，共 9,372 次正式解答；独立 32 题 × 4 工程试跑。只改变 8 个示范答案形式与新增的直接答案指令，其他设置见 experiment.json。baseline 是上一轮只读快照，禁止改写。新旧差值解释为这套提示的整体效果；未输出过程不能说明内部没有推理。

服务器目录：`/workspace/direct-answer`

推理解释器：`/venv/inference/bin/python`

报告环境：本目录 `.report-env`；独立安装 numpy==1.26.4、matplotlib==3.9.4、pyarrow==21.0.0。

从服务器目录用上述推理解释器执行 `-B run_all.py`。它依次执行输入核验、全部试跑、冻结、正式推理、实际恢复验证、统计、配对比较、独立审计、中文报告。以脱离 SSH 的进程运行即可无人值守完成。`.workflow.lock` 防止同一工作流并行，`/workspace/.gpu0.lock` 串行使用 GPU。失败见 `status/workflow.json` 和 `logs/`；修复基础设施后同命令恢复，只补缺失记录。已冻结的输入、配置和核心代码禁止修改。

结果入口 `reports/report_zh.md`（完整工作空间可生成 HTML）；原始回答在 `outputs/`，逐题 CSV、图表、统计及审计结果在 `reports/`，恢复验证在 `manifests/resume_check.json`。`status/workflow.json` 只有完整生成报告后才标为 complete。格式不符合与截断均保留计分，无正确率或格式通过门槛，不据试跑调整提示。

个人服务器同步工具留在旧本地工作空间，未随公开代码发布；本报告说明历史流程，不要求读者连接原服务器。
