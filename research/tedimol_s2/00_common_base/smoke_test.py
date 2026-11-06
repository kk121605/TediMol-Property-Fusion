"""Bounded pipeline check, full model on GPU / tiny model only on local CPU."""
import argparse
from datetime import datetime
import subprocess
import sys
from common import ROOT,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu-tiny',action='store_true')
    parser.add_argument('--microbatch',type=int,default=2)
    parser.add_argument('--sample-batch-size',type=int,default=8)
    args=parser.parse_args()
    if args.microbatch<1 or args.sample_batch_size<1:
        parser.error('Positive batch sizes required')
    device='cpu' if args.cpu_tiny else 'cuda'
    label='cpu_tiny' if args.cpu_tiny else 'gpu_full'
    train_limit=8 if args.cpu_tiny else 16
    test_limit=4 if args.cpu_tiny else max(8,args.sample_batch_size)
    batch=2 if args.cpu_tiny else 16
    micro=1 if args.cpu_tiny else args.microbatch
    sample_batch=2 if args.cpu_tiny else args.sample_batch_size
    run=ROOT/'runs'/('smoke_'+label+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    commands=[['preflight.py']+(['--cpu-check'] if args.cpu_tiny else []),
        ['process_text.py','-i','train_ChemBI','--limit',str(train_limit),'--device',device],
        ['process_text.py','-i','test_ChemBI','--limit',str(test_limit),'--device',device],
        ['train_Chem.py','--run',str(run),'--steps','2','--cache-limit',str(train_limit),'--batch-size',str(batch),
         '--microbatch',str(micro),'--save-every','1','--device',device]+(['--tiny'] if args.cpu_tiny else []),
        ['text_sample.py','--checkpoint',str(run/'latest.pt'),'--output',str(run/'generated.csv'),
         '--cache-limit',str(test_limit),'--limit',str(test_limit),'--batch-size',str(sample_batch),
         '--sample-steps','10','--device',device],
        ['evaluate.py','--input',str(run/'generated.csv'),'--allow-subset']]
    for cmd in commands:
        print('RUN:',sys.executable,*cmd,flush=True)
        subprocess.run([sys.executable]+cmd,cwd=ROOT,check=True)
    write_json(run/'smoke_receipt.json',dict(status='PASS',type=label,not_scientific_results=True,
        vocab_size=323,train_batch=batch,microbatch=micro,sample_batch=sample_batch,
        sample_length=258,training_steps=2,sampling_steps=10,full_200step_sampling_unverified=True))
    print(f'SMOKE PASS ({label}): {run}',flush=True)


if __name__=='__main__':
    main()
