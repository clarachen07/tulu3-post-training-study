"""New-versus-old tables and figures, consumed by the main report."""
import html
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT, load_json

NAMES = {'base':'Base', 'sft':'SFT', 'dpo':'DPO', 'rlvr':'RLVR'}


def table(headers, rows):
    md = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    md += ['| ' + ' | '.join(row) + ' |' for row in rows]
    h = '<div class="table"><table><thead><tr>' + ''.join('<th>'+html.escape(x)+'</th>' for x in headers) + '</tr></thead><tbody>'
    h += ''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
    return '\n'.join(md), h + '</tbody></table></div>'


def build():
    out = ROOT / 'reports'
    c = load_json(out / 'comparison.json')
    intro = ('差值方向统一为“直接答案版减上一轮”。本轮同时改变示范答案形式和输出指令，差值描述这套直接答案提示的整体效果；'
             '未展示计算过程不表示模型内部没有推理。每组使用相同题目，按题配对 bootstrap 10,000 次，报告百分位 95% 区间。')
    md = ['## 与上一轮逐题配对比较', '', intro, '']
    h = '<h2>与上一轮逐题配对比较</h2><p>'+intro+'</p>'
    rows, lengths = [], []
    for r in c['pairs']:
        split = '训练' if r['split'] == 'train' else '测试'
        rows.append([split, NAMES[r['stage']], f"{r['old_accuracy']*100:.2f}%", f"{r['new_accuracy']*100:.2f}%",
                     f"{r['delta']*100:+.2f}", f"[{r['ci_low']*100:+.2f}, {r['ci_high']*100:+.2f}]",
                     str(r['wrong_to_correct']), str(r['correct_to_wrong']), str(r['both_correct']), str(r['both_wrong'])])
        lengths.append([split, NAMES[r['stage']], f"{r['old_answer_only_rate']*100:.2f}% → {r['new_answer_only_rate']*100:.2f}%",
                        f"{r['old_mean_output_tokens']:.1f} → {r['new_mean_output_tokens']:.1f}",
                        f"{r['old_median_output_tokens']:.1f} → {r['new_median_output_tokens']:.1f}",
                        f"{r['old_truncation_rate']*100:.2f}% → {r['new_truncation_rate']*100:.2f}%"])
    a,b = table(['数据','阶段','上轮正确率','本轮正确率','差/百分点','95%区间/百分点','改善','退化','一直对','一直错'], rows)
    md += [a, '', '![新旧提示正确率差](protocol_delta.png)', '', '### 输出格式与长度（上轮 → 本轮）', '']
    h += b + '<img src="protocol_delta.png" alt="直接答案版减上一轮正确率及配对区间"><h3>输出格式与长度（上轮 → 本轮）</h3>'
    a,b = table(['数据','阶段','格式符合率','平均token','中位token','截断率'], lengths)
    md += [a, '']; h += b
    timing = [[NAMES[r['stage']], f"{r['old_process_seconds']/60:.2f} → {r['new_process_seconds']/60:.2f}",
               f"{r['old_generation_seconds']/60:.2f} → {r['new_generation_seconds']/60:.2f}"] for r in c['timing']]
    a,b = table(['阶段','正式进程耗时/分：上轮 → 本轮','生成耗时/分：上轮 → 本轮'], timing)
    md += [a, '', '计时为各阶段训练与测试合计；进程耗时含模型加载，受缓存及系统负载影响，不能完全归因于提示变化。', '',
           '### 同模型改善与退化案例', '', '[逐题对照](comparison_per_question.csv) · [完整案例](comparison_examples.json) · [对照统计](protocol_comparison.csv)。每阶段、每数据组、每方向按题目 ID 排序选前两例；网页可展开原始回答。', '']
    h += b + '<p>计时为各阶段训练与测试合计；进程耗时含模型加载，受缓存及系统负载影响，不能完全归因于提示变化。</p><h3>同模型改善与退化案例</h3><p>每阶段、每数据组、每方向按题目 ID 排序选前两例。</p>'
    for case in load_json(out / 'comparison_examples.json'):
        title = f"{case['split']} · {NAMES[case['stage']]} · {case['id']} · " + ('改善' if case['direction']=='wrong_to_correct' else '退化')
        h += '<details><summary>'+html.escape(title)+'</summary><p>'+html.escape(case['question'])+'</p><p>标准答案：'+html.escape(str(case['gold']))+'</p>'
        for key, label in [('old','上一轮'), ('new','直接答案版')]:
            r=case[key]
            h += '<h4>'+label+' · 提取数值 '+html.escape(str(r['extracted_answer']))+'</h4><pre>'+html.escape(r['continuation'])+'</pre>'
        h += '</details>'
    fig, ax=plt.subplots(figsize=(8,5), constrained_layout=True)
    for offset, split, color in [(-.12,'train','#237a78'),(.12,'test','#cc863c')]:
        rs=[r for r in c['pairs'] if r['split']==split]
        v=np.array([r['delta']*100 for r in rs]); lo=np.array([r['ci_low']*100 for r in rs]); hi=np.array([r['ci_high']*100 for r in rs])
        ax.errorbar(v, np.arange(4)+offset, xerr=np.maximum(0,[v-lo,hi-v]), fmt='o',capsize=4,color=color,label=split)
    ax.axvline(0,color='gray',lw=1); ax.set_yticks(np.arange(4), list(NAMES.values())); ax.invert_yaxis()
    ax.set_xlabel('Answer-only minus original accuracy (percentage points)'); ax.set_title('Paired question bootstrap: 95% intervals'); ax.legend(frameon=False); ax.grid(axis='x',alpha=.15)
    fig.savefig(out/'protocol_delta.png',dpi=220); fig.savefig(out/'protocol_delta.svg'); plt.close(fig)
    return '\n'.join(md), h
