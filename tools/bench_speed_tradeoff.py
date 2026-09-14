#!/usr/bin/env python3
"""Sequential C/C warm A/B: retention or prepared-reference, with full provenance.

Only one inference worker is resident at a time. Every timed sample has its own
fresh process and warm-up. This is not the simultaneous Python/C P3 protocol.
Memory instrumentation runs separately via --memory-profile; never accepted as
latency evidence. No shell interpolation or system tuning is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time
import wave

from bench_four_modes import wait_for_idle, validate_worker_ready

ROOT = Path(__file__).resolve().parents[1]
TEXT = 'こんにちは、色とりどりの世界へようこそ。'
CAPTION = '落ち着いた自然な女性の声で、やわらかく話す。'
GOLDEN = {'text-only':'seed42_steps8', 'caption-only':'caption_seed42_steps8',
          'clone':'kana_clone_seed42_steps8', 'clone+caption':'kana_clone_caption_seed42_steps8'}

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def swaps():
    return {k:int(v) for k,v in (l.split() for l in Path('/proc/vmstat').read_text().splitlines()) if k in ('pswpin','pswpout')}

def percentile(xs, q):
    a=sorted(xs); i=(len(a)-1)*q; lo=int(i); hi=min(lo+1,len(a)-1)
    return a[lo]+(a[hi]-a[lo])*(i-lo)

class Worker:
    def __init__(self,cmd,env,log,timeout=900):
        self.log=log.open('w'); self.lines=queue.Queue(); self.timeout=timeout
        try:
            self.p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        except Exception:
            self.log.close();raise
        self.reader=threading.Thread(target=self.read,daemon=True);self.reader.start()
    def read(self):
        try:
            for line in self.p.stdout:
                self.log.write(line);self.log.flush();self.lines.put(line)
        finally:self.lines.put(None)
    def until(self,prefix):
        deadline=time.monotonic()+self.timeout
        while True:
            try:l=self.lines.get(timeout=max(.001,deadline-time.monotonic()))
            except queue.Empty:raise RuntimeError('worker timeout')
            if l is None:raise RuntimeError(f'worker exited: {self.p.poll()}')
            if l.startswith(prefix):return json.loads(l[len(prefix):])
    def request(self,cmd):
        self.p.stdin.write(cmd+'\n');self.p.stdin.flush()
        return self.until('__IRO_RESULT__=')
    def close(self):
        try:
            if self.p.poll() is None:
                self.p.stdin.write('QUIT\n');self.p.stdin.flush();self.p.wait(timeout=10)
        except (BrokenPipeError,subprocess.TimeoutExpired):
            self.p.kill();self.p.wait()
        finally:
            self.reader.join(timeout=5)
            for stream in (self.p.stdin,self.p.stdout,self.log):
                try:stream.close()
                except BrokenPipeError:pass

def summarize(rows,pairs,profile):
    result={}
    for mode,step in sorted({(r['mode'],r['steps']) for r in rows}):
        selected=[r for r in rows if r['mode']==mode and r['steps']==step]
        groups={v:[r for r in selected if r['variant']==v] for v in ['baseline','candidate']}
        if any(len(g)!=pairs for g in groups.values()):continue
        b,c=groups['baseline'],groups['candidate']
        ratios=[next(r['run']['elapsed_seconds'] for r in b if r['pair']==i)/next(r['run']['elapsed_seconds'] for r in c if r['pair']==i) for i in range(1,pairs+1)]
        out={'paired_speedups':ratios,'all_candidate_faster':all(x>1 for x in ratios),'wav_exact':len({r['wav_sha256'] for r in selected})==1}
        for v,rs in groups.items():
            times=[r['run']['elapsed_seconds'] for r in rs]
            out[v]={'p50_seconds':statistics.median(times),'p95_seconds':percentile(times,.95),
                    'peak_rss_mib':max(r['run']['peak_rss_kib'] for r in rs)/1024,
                    'minor_faults_median':statistics.median(r['run']['minor_faults'] for r in rs),
                    'stages_seconds':{k:statistics.median(r['run'][k] for r in rs) for k in ['encode_seconds','sample_seconds','decode_seconds']},
                    'rtf':statistics.median(times)/rs[0]['audio_seconds']}
        out['speedup']=out['baseline']['p50_seconds']/out['candidate']['p50_seconds']
        out['clean']=all(r['idle_warm']['passed'] and r['idle_run']['passed'] and not any(r['swap_delta'].values()) and r['run']['major_faults']==0 for r in selected)
        out['latency_candidate_gate']=bool(pairs>=5 and not profile and out['clean'] and out['wav_exact'] and out['all_candidate_faster'] and out['speedup']>=1/.98 and out['candidate']['p95_seconds']<=out['baseline']['p95_seconds']*1.05)
        result[f'{mode}-s{step}']=out
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--worker',type=Path,default=ROOT/'irodori-bench-worker-mkl')
    ap.add_argument('--backend',default='torch-mkl-sgemm')
    ap.add_argument('--reference',type=Path,default=Path('/home/kenobu/development/kana-alya/assets/voices/kana-default.wav'))
    ap.add_argument('--modes',default='text-only')
    ap.add_argument('--steps',default='8')
    ap.add_argument('--pairs',type=int,default=3)
    ap.add_argument('--experiment',choices=['retention','reference','packed','int8','codec-int8'],default='retention')
    ap.add_argument('--packed-mib',type=int,default=512)
    ap.add_argument('--dump-dir',type=Path,help='Diagnostic trace directory per variant/sample; timings not acceptance')
    ap.add_argument('--memory-profile',action='store_true')
    ap.add_argument('--allow-busy',action='store_true')
    ap.add_argument('--idle-wait',type=float,default=15)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out=args.out.absolute()
    modes=args.modes.split(',');steps=[int(x) for x in args.steps.split(',')]
    if not 0<=args.packed_mib<=2048:ap.error('packed MiB must be 0..2048')
    if args.pairs<1 or any(s<=0 for s in steps) or any(m not in GOLDEN for m in modes):ap.error('invalid pairs, steps or modes')
    if args.experiment=='reference' and any(not m.startswith('clone') for m in modes):ap.error('reference requires clone modes')
    args.out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update({k:'2' for k in ['IRO_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']})
    # int8 needs a VNNI-capable MKL branch; AVX2/AVX512 CBWR halve its throughput. FP32 speed is
    # branch-independent on this host, so both variants of the int8 experiment use AVX512_E1.
    if args.backend=='onemkl-sgemm':env.setdefault('MKL_CBWR','AVX512_E1' if args.experiment in ('int8','codec-int8') else 'AVX2')
    for key in ['IRO_DIT_PROFILE','IRO_GEMM_PROFILE','IRO_DACVAE_PROFILE','IRO_ENCODER_PROFILE','IRO_DACVAE_OP_PROFILE']:env.pop(key,None)
    paths=[args.worker.absolute(),Path(str(args.worker.absolute())+'.bin'),args.model,ROOT/'generate.c',ROOT/'generate.h',ROOT/'ops.c',ROOT/'ops.h',ROOT/'dit.c',ROOT/'dit.h',ROOT/'sampler.c',ROOT/'Makefile',Path(__file__).absolute(),ROOT/'tools/bench_c_worker.c',ROOT/'weights/tokenizer.bin',ROOT/'weights/dacvae_decoder.safetensors']
    if any(m.startswith('clone') for m in modes):paths.extend([args.reference,ROOT/'weights/dacvae_encoder.safetensors'])
    paths.extend(ROOT/'golden'/GOLDEN[m]/('x_t_step000.f32' if m=='text-only' else 'noise.f32') for m in modes)
    if any(not p.is_file() for p in paths if not str(p).endswith('.bin')):ap.error('missing required input; check worker/model/noise assets')
    report={'status':'running','protocol':'sequential fresh process + warm-up before each measured sample; alternating first variant',
            'argv':sys.argv,'experiment':args.experiment,'memory_profile':args.memory_profile or bool(args.dump_dir),'pairs':args.pairs,
            'cpus':[0,1],'threads':2,'backend':args.backend,'extra_ram_budget_mib':1536 if args.experiment=='retention' else args.packed_mib if args.experiment=='packed' else 512,'quality_note':'int8 candidate PCM differs from baseline by design; see tools/int8_quality_corpus.py' if args.experiment in ('int8','codec-int8') else None,'inherited_env':{k:env.get(k,'') for k in ['IRO_DIT_PRECISION','IRO_CODEC_PRECISION']},
            'numeric_quality_gate':'separate artifact required for packed; this runner never accepts drift on its own',
            'environment':{k:env.get(k,'') for k in ['MKL_CBWR','IRO_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},
            'hashes':{str(p):digest(p) for p in paths if p.is_file()},'rows':[]}
    def save():
        report['summary']=summarize(report['rows'],args.pairs,args.memory_profile or bool(args.dump_dir))
        tmp=args.out/'summary.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(args.out/'summary.json')
    def idle():
        x=wait_for_idle(minimum_fraction=.9,sample_seconds=1.,timeout_seconds=args.idle_wait)
        if not x['passed'] and not args.allow_busy:raise RuntimeError(f"host not idle: {x['idle_fraction']:.3f}")
        return x
    try:
        save()
        for mode in modes:
            for step in steps:
                expected_hash=None
                for pair in range(1,args.pairs+1):
                    for variant in (['baseline','candidate'] if pair%2 else ['candidate','baseline']):
                        stem=f'{mode}-s{step}-p{pair}-{variant}';wav=args.out/(stem+'.wav')
                        cmd=['taskset','-c','0,1',str(args.worker.absolute()),'--model',str(args.model.absolute()),'--tokenizer',str(ROOT/'weights/tokenizer.bin'),'--decoder',str(ROOT/'weights/dacvae_decoder.safetensors'),'--text',TEXT,'--noise',str(ROOT/'golden'/GOLDEN[mode]/('x_t_step000.f32' if mode=='text-only' else 'noise.f32')),'--seed','42','--steps',str(step),'--memory-profile',str(int(args.memory_profile))]
                        if mode.startswith('clone'):cmd+=['--encoder',str(ROOT/'weights/dacvae_encoder.safetensors'),'--ref',str(args.reference.absolute())]
                        if 'caption' in mode:cmd+=['--caption',CAPTION]
                        flag={'retention':'--retain-frontend','reference':'--prepared-reference','packed':'--packed-mib','int8':'--dit-precision','codec-int8':'--codec-precision'}[args.experiment]
                        if args.experiment in ('int8','codec-int8'):value='int8' if variant=='candidate' else 'fp32'
                        else:value=(args.packed_mib if args.experiment=='packed' else 1) if variant=='candidate' else 0
                        cmd += [flag,str(value)]
                        if args.dump_dir:cmd += ['--dump-dir',str(args.dump_dir.absolute()/stem)]
                        iw=idle(); before=swaps(); begin=time.monotonic();worker=Worker(cmd,env,args.out/(stem+'.log'))
                        try:
                            ready=worker.until('__IRO_READY__=');startup=time.monotonic()-begin
                            validate_worker_ready(stem,ready,2,args.backend)
                            if ready[{'retention':'retain_frontend','reference':'prepared_reference','packed':'packed_mib','int8':'dit_precision','codec-int8':'codec_precision'}[args.experiment]]!=value:raise RuntimeError('variant mismatch')
                            warm=worker.request('WARM\t'+str(wav));warmhash=digest(wav)
                            ir=idle();run=worker.request('RUN\t'+str(wav));h=digest(wav)
                            # int8 deliberately changes the DiT arithmetic: WARM/RUN must still be byte-identical
                            # per variant, but baseline and candidate PCM are expected to differ.
                            if h!=warmhash or (args.experiment not in ('packed','int8','codec-int8') and expected_hash is not None and expected_hash!=h):raise RuntimeError('PCM byte parity failed')
                            if args.experiment in ('int8','codec-int8'):
                                key=('int8hash',mode,step,variant)
                                if report.setdefault('variant_hashes',{}).get(str(key),h)!=h:raise RuntimeError('int8 variant PCM not deterministic across pairs')
                                report['variant_hashes'][str(key)]=h
                            expected_hash=h
                            if any(not math.isfinite(run[k]) or run[k]<0 for k in ['elapsed_seconds','encode_seconds','sample_seconds','decode_seconds']):raise RuntimeError('invalid timing')
                            with wave.open(str(wav)) as f:
                                if f.getnframes()!=run['output_samples'] or f.getnframes()<=0:raise RuntimeError('invalid WAV')
                                duration=f.getnframes()/f.getframerate()
                        finally:worker.close()
                        after=swaps()
                        row={'mode':mode,'steps':step,'pair':pair,'variant':variant,'command':cmd,'ready':ready,'startup_seconds':startup,'warm':warm,'run':run,'idle_warm':iw,'idle_run':ir,'swap_delta':{k:after[k]-before[k] for k in before},'wav_sha256':h,'wav':str(wav),'audio_seconds':duration}
                        report['rows'].append(row);save()
                        print(f"{stem}: {run['elapsed_seconds']:.3f}s RSS={run['peak_rss_kib']/1024:.1f}MiB faults={run['minor_faults']} idle={ir['idle_fraction']:.1%}",flush=True)
        report['status']='complete';save();return 0
    except Exception as e:
        report['status']='aborted';report['error']=str(e);save();raise

if __name__=='__main__':raise SystemExit(main())
