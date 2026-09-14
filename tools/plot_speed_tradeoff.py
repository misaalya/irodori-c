#!/usr/bin/env python3
"""Render recorded C/C experiments; never fabricate absent measurements."""
import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory.absolute()
 rows=[]
 for name in ['retention-diagnostic-v2','packed-e2e','reference-s8']:
  f=root/name/'summary.json'
  if not f.exists():continue
  d=json.loads(f.read_text())
  for mode,r in d.get('summary',{}).items():rows.append((name,mode,r))
 if rows:
  fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
  colors=['#607d8b','#00897b'];xs=list(range(len(rows)))
  for j,v in enumerate(['baseline','candidate']):
   for ax,key,label in [(axes[0],'p50_seconds','Warm latency p50 (s)'),(axes[1],'peak_rss_mib','Lifetime peak RSS (MiB)')]:
    values=[r[v][key] for _,_,r in rows]
    bars=ax.bar([x+(j-.5)*.36 for x in xs],values,.36,label=v.title(),color=colors[j])
    ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9);ax.set_ylabel(label)
    ax.set_xticks(xs,[name.replace('-diagnostic-v2','').replace('-e2e','')+'\n'+mode for name,mode,_ in rows]);ax.set_ylim(0,max(r[engine][key] for _,_,r in rows for engine in ['baseline','candidate'])*1.22)
  for ax in axes:ax.legend();ax.spines[['top','right']].set_visible(False)
  fig.suptitle('C/C speed–RAM experiments • diagnostic, not final acceptance')
  fig.savefig(root/'speed-ram.png',dpi=150);plt.close(fig)
 f=root/'packed-screen.jsonl'
 if f.exists():
  data=[json.loads(x) for x in f.read_text().splitlines()];shapes=sorted({r['M'] for r in data})
  fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
  for i,m in enumerate(shapes):
   values=[r['speedup'] for r in data if r['M']==m and r['type']=='packed_pair']
   ax.scatter([i]*len(values),values,color='#00897b',alpha=.6)
   ax.plot([i-.2,i+.2],[statistics.median(values)]*2,color='#263238',lw=3)
  ax.axhline(1,color='#b71c1c',ls='--');ax.set_xticks(range(len(shapes)),shapes)
  ax.set(xlabel='GEMM M (rows)',ylabel='Baseline / packed time',title='W1 packed GEMM • five paired kernel measurements\nPacking cost excluded here; included in first inference')
  fig.savefig(root/'packed-kernel.png',dpi=150);plt.close(fig)
 # Stage snapshots, intentionally no interpolated transient-peak claims.
 logs=list((root/'memory-stages').glob('*.log')) if (root/'memory-stages').exists() else []
 if logs:
  fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
  for log in logs:
   vals=[json.loads(l.split('=',1)[1]) for l in log.read_text().splitlines() if l.startswith('__IRO_MEMORY__=')]
   if not vals:continue
   ax.plot(range(len(vals)),[v['rss_kib']/1024 for v in vals],marker='o',label=log.stem.split('-')[-1])
   ax.set_xticks(range(len(vals)),[v['stage'] for v in vals],rotation=35,ha='right')
  ax.set(ylabel='RSS (MiB)',title='Stage-boundary snapshots • warm-up then measured request\nTransient intra-stage peaks are not sampled');ax.legend()
  fig.savefig(root/'memory-stages.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
