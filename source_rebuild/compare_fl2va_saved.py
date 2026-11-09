"""Use the unmodified official comparator with competition-provided dependencies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tools.build_outputs import sha
from tools.platform_environment import check

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-video',type=Path,required=True)
    parser.add_argument('--reference-video',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    check(ROOT,'metrics')
    config=json.loads((ROOT/'config/run.json').read_text(encoding='utf-8'))
    script=ROOT/'runtime/official_compare_reference.py'
    if sha(script)!=config['official_comparator_sha256']:raise RuntimeError('Official comparator source changed')
    command=[sys.executable,str(script),'--tasks','fl2va','--model-path',config['model_path'],
             '--reference-fl2va-video',str(args.reference_video.resolve()),'--infinilm-fl2va-video',str(args.candidate_video.resolve()),
             '--output-dir',str(args.output_dir.resolve())]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    (args.output_dir/'official_compare.stdout.log').write_text(result.stdout,encoding='utf-8')
    (args.output_dir/'official_compare.stderr.log').write_text(result.stderr,encoding='utf-8')
    (args.output_dir/'comparison_provenance.json').write_text(json.dumps(dict(comparator_sha256=sha(script),comparator_unmodified=True,
        reference_sha256=sha(args.reference_video),candidate_sha256=sha(args.candidate_video),returncode=result.returncode,new_model_or_gpu_execution=False),indent=2)+'\n')
    print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
