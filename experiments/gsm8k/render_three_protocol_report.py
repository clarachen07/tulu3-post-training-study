#!/usr/bin/env python3
"""Render English three-condition figures and report from retained statistics."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
STAGES = ('base','sft','dpo','rlvr')
GROUPS = {'cot':'Reasoning demonstrations','direct-answer':'Direct-answer demonstrations','cot-direct-answer':'Reasoning demonstrations + direct-answer instruction'}
COLORS = ('#176D88','#B95F2B','#7254A3')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root):
    source=root/'cot-direct-answer/verify_publication.py'
    spec=importlib.util.spec_from_file_location('gsm8k_publication_check',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify(root/'cot-direct-answer')
    stats={g:read(root/g/'reports/statistics.json') for g in GROUPS}
    for group in GROUPS:
        audit=read(root/group/'reports/independent_audit.json')
        if audit['status']!='passed' or sha(root/group/'reports/statistics.json')!=audit['statistics_sha256']:
            raise ValueError('Historical statistics identity differs: '+group)
    return stats


def build(root=ROOT, output_dir=None):
    root=Path(root).resolve();stats=load(root)
    destination=Path(output_dir).resolve() if output_dir else root
    reports=destination/'reports';figures=reports/'figures';figures.mkdir(parents=True,exist_ok=True)
    lookup={g:{(r['split'],r['stage']):r for r in s['summary']} for g,s in stats.items()}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.dpi':250})
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.2),sharey=True)
    fig.subplots_adjust(left=.075,right=.975,bottom=.20,top=.72,wspace=.12)
    fig.suptitle('GSM8K: demonstration format and direct-answer instruction',x=.075,y=.97,ha='left',fontsize=17,weight='bold')
    plotted=[]
    for ax,split in zip(axes,('test','train')):
        for (group,label),color,marker,offset in zip(GROUPS.items(),COLORS,('o','s','D'),(-.05,0,.05)):
            rows=[lookup[group][split,s] for s in STAGES]
            means=[100*r['accuracy'] for r in rows]
            errors=[[100*(r['accuracy']-r['ci_low']) for r in rows],[100*(r['ci_high']-r['accuracy']) for r in rows]]
            ax.errorbar([i+offset for i in range(4)],means,yerr=errors,color=color,marker=marker,linewidth=2,capsize=3,label=label)
            plotted.extend({'protocol':group,**{k:r[k] for k in ('split','stage','n','correct','accuracy','ci_low','ci_high')}} for r in rows)
        ax.set_title(('Full test set' if split=='test' else 'Training-pool sample')+f" (n = {1319 if split=='test' else 1024:,})",loc='left')
        ax.set_xticks(range(4),('Base','SFT','DPO','RLVR'));ax.set_xlim(-.18,3.18);ax.set_ylim(0,100)
        ax.set_yticks(range(0,101,20),[f'{v}%' for v in range(0,101,20)]);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Final numerical-answer accuracy')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.07,.89),frameon=False,ncol=1,fontsize=9)
    fig.text(.075,.09,'Error bars: question-level bootstrap 95% intervals. Historical conditions were not rerun.',fontsize=10)
    fig.text(.075,.045,'Direct-answer instruction compliance: 0.00%-0.23% on test; numerical accuracy includes noncompliant responses.',fontsize=10)
    outputs=[]
    for ext in ('png','svg'):
        path=figures/f'four-stage-accuracy.{ext}';fig.savefig(path);outputs.append(path)
    plt.close(fig)
    path=reports/'three_protocol_figure_data.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(plotted[0]));writer.writeheader();writer.writerows(plotted)
    outputs.append(path)
    lines=['# GSM8K: three matched prompting conditions','','Compare frozen Base, SFT, DPO and RLVR checkpoints on the same 1,024 training-pool and 1,319 test questions. The third condition prepends the existing direct-answer instruction and two newlines to the unchanged eight reasoning demonstrations.','','## Results','','| Split | Stage | Reasoning demos | Direct-answer demos | Reasoning demos + direct instruction | Third-condition compliance |','|---|---|---:|---:|---:|---:|']
    for split in ('test','train'):
        for stage in STAGES:
            scores=[f"{100*lookup[g][split,stage]['accuracy']:.2f}%" for g in GROUPS]
            compliance=lookup['cot-direct-answer'][split,stage]['answer_only_rate']
            lines.append('| '+' | '.join([split,stage.upper(),*scores,f'{100*compliance:.2f}%'])+' |')
    lines+=['','![Three-condition accuracy](figures/four-stage-accuracy.png)','','The direct-answer instruction was rarely followed. Third-condition numerical accuracy therefore does not measure accuracy under successfully enforced answer-only output. Correct-and-compliant rates are zero for every test-stage cell.','','## Protocol and limits','','All three conditions use greedy BF16 decoding, a 1,024-token output budget, an 8,192-token context and the historical last-number scorer. Base uses plain text; tuned checkpoints use native chat templates. The third condition changes the instruction while retaining reasoning demonstrations. Only that condition was newly generated.','','Question-paired percentile bootstrap intervals use 10,000 draws. Retained per-question scores support independent reconstruction of counts, stage changes and all 16 historical contrasts; original text is needed to rescore answers and format compliance. Intervals describe sensitivity to sampled questions, without multiple-comparison correction or repeated-training uncertainty.','','Absent emitted reasoning does not establish absent internal reasoning. Demonstration changes also change input length. Native wrappers and frozen-checkpoint comparisons do not isolate training algorithms or prove training exposure.','','The Chinese reports remain separate historical narrative artifacts. This renderer rebuilds the English report and the single canonical English PNG/SVG figure from retained results.']
    path=reports/'report.md';path.write_text('\n'.join(lines)+'\n');outputs.append(path)
    manifest={'scope':'English publication rendering from verified retained score statistics; no inference or raw-answer rescoring','inputs':{f'{g}/reports/statistics.json':sha(root/g/'reports/statistics.json') for g in GROUPS},'outputs':{str(p.relative_to(destination)):sha(p) for p in outputs},'renderer_sha256':sha(Path(__file__)),'summary_cells':24,'historical_paired_contrasts':16}
    (reports/'three_protocol_report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Rendered 24 summary cells and two canonical English figures; score-derived intervals verified.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--font',help='Accepted for compatibility; English figures use DejaVu Sans')
    args=parser.parse_args();build(args.root,args.output_dir)


if __name__=='__main__':
    main()
