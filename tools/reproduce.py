import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
import traceback
import uuid
import zipfile

REPO = Path(os.environ['MINIMAX_REPRO_ROOT']).resolve()
PACKAGE = 'preliminary_dependencies.zip'
EXPECTED = 'bfcf7b5b629938549dab6303c8fe3c249d8a996f313b9f3048c5f20004409098'
AFS_DEFAULT = str(REPO / 'work')
started = time.monotonic()
deadline = started + 1740
parser = argparse.ArgumentParser()
parser.add_argument('--first-frame', type=Path, default=REPO / 'assets/Golde33443.jpg')
parser.add_argument('--model-path', type=Path, default=Path(os.environ.get('MODEL_PATH', '/public/home/aclsylqidf/shared/MiniMax-H3')))
parser.add_argument('--reference-video', type=Path, default=Path(os.environ['REFERENCE_VIDEO']) if os.environ.get('REFERENCE_VIDEO') else REPO / 'examples/reference_fl2va.mp4')
parser.add_argument('--from-source', action='store_true', default=True,
                    help='Build from submitted source before inference; this is the default.')
parser.add_argument('--build-only', action='store_true')
parser.add_argument('--jobs', type=int, default=6)
parser.add_argument('--verify-only', action='store_true', help='CPU verification on the competition platform; default runs the full eight-card case.')
args = parser.parse_args()
afs = Path(os.environ.get('IFAB_AFS_HOME') or AFS_DEFAULT).resolve()
run_id = datetime.now(timezone.utc).strftime('preliminary_%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
output = afs / 'minimax_preliminary_runs' / run_id
output.mkdir(parents=True, exist_ok=False)
print('REPRO_OUTPUT_DIRECTORY', output, flush=True)
print('REPRO_MODE', 'SOURCE_BUILD_AND_CPU_BINDINGS' if args.verify_only else 'SOURCE_BUILD_THEN_EIGHT_CARD_FULL_FL2VA', flush=True)
report = dict(version='preliminary_source_default', mode='source_build_with_private_dependencies',
              result_directory_id=run_id, platform_run_id=None,
              utc=datetime.now(timezone.utc).isoformat(), cpu_only=args.verify_only,
              status='STARTING', actual_model_executed=False, preliminary_admission_passed=False,
              source_only_submission=True, compiler_invoked=False, pip_invoked=False,
              model_weights_included=False, model_deadline_seconds=1740, job_budget_seconds=1800,
              source_build_required=True, packaged_project_binaries_used=False,
              scheduler_identifiers={k: os.environ[k] for k in ('SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID', 'IFAB_JOB_ID') if os.environ.get(k)})
scratch = None
child = None

class Checkpoint(Exception):
    pass

def extract_build(package, folder):
    with zipfile.ZipFile(package) as archive, archive.open('build.tar.xz') as stream, tarfile.open(fileobj=stream, mode='r|xz') as t:
        written = set()
        for item in t:
            p = PurePosixPath(item.name)
            if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'build':
                raise RuntimeError('Unsafe build dependency path')
            target = folder.joinpath(*p.parts)
            if item.islnk():
                q = PurePosixPath(item.linkname)
                if item.linkname not in written or q.is_absolute() or '..' in q.parts or q.parts[0] != 'build':
                    raise RuntimeError('Unsafe build dependency hardlink')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(folder.joinpath(*q.parts), target)
            elif item.isfile() or item.isdir():
                t.extract(item, folder, filter='data')
            else:
                raise RuntimeError('Unexpected build dependency object')
            if item.isfile() or item.islnk():
                written.add(item.name)
    return folder / 'build'

def wait_process(command, env, name, limit):
    global child
    with (output / name).open('wb') as log:
        child = subprocess.Popen(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        end = min(deadline, time.monotonic() + limit)
        with (output / name).open('r', encoding='utf-8', errors='replace') as events:
            last_heartbeat = 0
            while True:
                for line in events.read().splitlines():
                    print(line, flush=True)
                code = child.poll()
                if code is not None:
                    for line in events.read().splitlines():
                        print(line, flush=True)
                    break
                if time.monotonic() >= end:
                    stop_tree()
                    raise TimeoutError('Phase deadline: ' + name)
                elapsed = time.monotonic() - started
                if elapsed - last_heartbeat >= 15:
                    last_heartbeat = elapsed
                    print('REPRO_PHASE_RUNNING', name, 'elapsed=' + str(round(elapsed, 1)), flush=True)
                time.sleep(1)
    return child.returncode

def save():
    (output / 'ready_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def phase(name):
    report['phase'] = name
    save()
    print('REPRO_PHASE', name, 'elapsed_seconds=' + str(round(time.monotonic() - started, 1)), flush=True)
    if time.monotonic() >= deadline:
        raise TimeoutError('1740-second runtime deadline reached; reserve time for the result ZIP')

def stop_tree():
    if child is None or child.poll() is not None:
        return
    descendants = {child.pid}
    # Stop only this launched supervisor and its children, including child sessions.
    for _ in range(12):
        old = set(descendants)
        for p in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = p.read_text().rsplit(')', 1)[1].split()
                if int(fields[1]) in descendants:
                    descendants.add(int(p.parent.name))
            except (OSError, ValueError, IndexError):
                pass
        if old == descendants:
            break
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in sorted(descendants, reverse=True):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(0.5)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

def interrupted(signum, frame):
    raise InterruptedError('Received signal ' + str(signum))

for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, interrupted)

exit_code = 2
try:
    phase('check_upload')
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 11):
        raise RuntimeError('Requires the organizer Linux/Python 3.11 DTK container')
    launcher = REPO / 'reproduce.sh'
    report['driver_sha256'] = digest(Path(__file__))
    report['launcher_sha256'] = digest(launcher)
    locations = [REPO / 'environment' / PACKAGE]
    package = next((p for p in locations if p.is_file()), None)
    if package is None:
        raise FileNotFoundError('Missing repository environment/' + PACKAGE + '; download the complete Git LFS object')
    report['package'] = str(package)
    with package.open('rb') as stream:
        pointer = stream.read(96).startswith(b'version https://git-lfs.github.com/spec/')
    if pointer:
        raise RuntimeError('Environment is a Git LFS pointer. Run git lfs pull in this repository before submitting the GPU job.')
    report['package_sha256'] = digest(package)
    if report['package_sha256'] != EXPECTED:
        raise RuntimeError('ZIP SHA differs. Use the matching repository environment archive and script; do not rename an old ZIP.')
    with zipfile.ZipFile(package) as z:
        if set(z.namelist()) != {'runtime.tar.xz', 'build.tar.xz', 'manifest.json'}:
            raise RuntimeError('Unexpected package members')
        meta = json.loads(z.read('manifest.json'))
        required = meta['unpacked_bytes'] + 512 * 1024**2
        candidates = [os.environ.get('TMPDIR'), str(Path.cwd()), '/tmp', '/dev/shm', str(afs)]
        attempts = []
        for name in dict.fromkeys(x for x in candidates if x):
            try:
                parent = Path(name).resolve()
                if not parent.is_dir() or shutil.disk_usage(parent).free < required:
                    raise OSError('Missing directory or insufficient free space')
                scratch = Path(tempfile.mkdtemp(prefix='minimax_preliminary_', dir=parent))
                (scratch / 'owned_preliminary').write_text(run_id)
                report['scratch_directory'] = str(scratch)
                break
            except OSError as e:
                attempts.append(dict(path=name, reason=str(e)))
        report['scratch_attempts'] = attempts
        if scratch is None:
            raise RuntimeError('No writable runtime directory with enough space: ' + str(attempts))
        phase('extract_private_python_dependencies')
        if meta.get('project_precompiled_artifacts_included') is not False or meta.get('official_base_image_included') is not False:
            raise RuntimeError('Expected the dependency-only source-build environment')
        extracted_files = set()
        with z.open('runtime.tar.xz') as stream, tarfile.open(fileobj=stream, mode='r|xz') as t:
            for member in t:
                path = PurePosixPath(member.name)
                allowed = member.name.startswith('runtime/site/') or member.name in {
                    'runtime', 'runtime/site', 'runtime/files.json',
                    'runtime/provenance.json', 'runtime/setup.json'}
                if path.is_absolute() or '..' in path.parts or '\\' in member.name or not allowed:
                    raise RuntimeError('Unexpected dependency-only runtime path')
                if not (member.isfile() or member.isdir()):
                    raise RuntimeError('Runtime dependencies must contain regular files or directories')
                if member.name in extracted_files:
                    raise RuntimeError('Duplicate dependency file')
                t.extract(member, scratch, filter='data')
                if member.isfile():
                    extracted_files.add(member.name)
                if time.monotonic() >= deadline:
                    raise TimeoutError('Runtime extraction exceeded the job deadline')
        report['environment_contains_project_binaries'] = False
        report['environment_contains_official_image'] = False
    root = scratch / 'runtime'
    phase('verify_private_python_dependencies')
    inventory = json.loads((root / 'files.json').read_text())
    if len(inventory['files']) != meta['files']:
        raise RuntimeError('Runtime inventory count differs')
    dependency_files = [item for item in inventory['files'] if item['path'].startswith('site/')]
    if not dependency_files:
        raise RuntimeError('Missing private dependency inventory')
    for item in dependency_files:
        p = root / item['path']
        if not p.resolve().is_relative_to(root.resolve()) or p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
            raise RuntimeError('Runtime file differs: ' + item['path'])
    report['verified_dependency_files'] = len(dependency_files)
    report['archive_inventory_records'] = len(inventory['files'])
    report['runtime_payload_sha256'] = meta['runtime_payload_sha256']
    for name in ('provenance.json', 'setup.json', 'files.json'):
        shutil.copyfile(root / name, output / name)
    if digest(args.first_frame) != '1e71abf29876f248088a11211a350ed3d26a0c68aa34f6e45738771eb2d45a22':
        raise RuntimeError('Use the original public-case first frame')
    report['model_path'] = str(args.model_path)
    env = os.environ.copy()
    # The source-run entry sets all seven native roots from rebuilt_assets and
    # verifies its bindings belong to this build. Do not inherit cached routes.
    for variable in list(env):
        if variable.startswith('INFINICORE_NINETOOTHED_') and variable.endswith('_ROOT'):
            env.pop(variable)
    for variable in ('INFINI_ROOT', 'MINIMAX_PORTABLE_ROOT', 'MINIMAX_PYTHON_RUNTIME_SITE',
                     'MINIMAX_PYTHON_AUX_ROOT', 'MINIMAX_RMS_PROBE_ROOT', 'MINIMAX_RMS_PROBE_PHASE'):
        env.pop(variable, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    env.update(PYTHONPATH=str(root / 'site'), PIP_NO_INDEX='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    source_mode = True
    if source_mode:
        report['source_build_requested'] = True
        report['compiler_invoked'] = None
        phase('source_build_environment')
        build_base = Path(os.environ.get('BUILD_WORK', str(afs / ('minimax_preliminary_source_' + EXPECTED[:12])))).resolve()
        build_base.mkdir(parents=True, exist_ok=True)
        work = build_base / 'outputs'
        report['source_build_resuming'] = (work / 'build_report.json').is_file()
        build_deps = extract_build(package, build_base / 'dependencies')
        tools = build_deps / 'tools'
        rc = build_deps / 'xmake-offline.lua'
        rc.write_text((tools / 'offline/xmake-extra.lua').read_text() + '\n' + (build_deps / 'openmp_recipe.lua').read_text() + '\npackage_end()\n')
        env.update(PYTHONPATH=str(build_deps / 'site') + ':' + str(root / 'site'),
                   PATH=str(tools / 'tools/bin') + ':' + env.get('PATH', ''),
                   MINIMAX_OFFLINE_ROOT=str(tools), XMAKE_RCFILES=str(rc), XMAKE_ROOT='y')
        for name in ('C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH'):
            env[name] = sysconfig.get_path('include') + ':' + env.get(name, '')
        report['source_build_workdir'] = str(work)
        for name, leaf in (('XMAKE_PKG_INSTALLDIR', 'packages'), ('XMAKE_PKG_CACHEDIR', 'downloads')):
            p = build_base / 'package_cache' / leaf
            p.mkdir(parents=True, exist_ok=True)
            env[name] = str(p)
        phase('compile_source')
        code = wait_process([sys.executable, REPO / 'source_rebuild/build.py', '--workdir', work, '--stage', 'all', '--jobs', args.jobs], env, 'source_build.log', max(1, deadline - time.monotonic() - 70))
        source_report = work / 'build_report.json'
        if source_report.exists():
            shutil.copyfile(source_report, output / 'build_report.json')
        if code == 75:
            raise Checkpoint('Source build saved progress; repeat the same source command in another job')
        if code:
            raise RuntimeError('Source build failed; return complete result ZIP')
        if args.build_only:
            report['source_build_complete'] = True
            raise Checkpoint('Source build completed; repeat bash reproduce.sh to run inference')
        if not args.verify_only and deadline - time.monotonic() < 650:
            raise Checkpoint('Build completed; insufficient inference budget remains. Repeat bash reproduce.sh.')
        phase('source_built_inference')
        command = [sys.executable, '-u', REPO / 'source_rebuild/run.py', '--from-build', work, '--first-frame', args.first_frame, '--model-path', args.model_path, '--env-root', output / 'source_run']
        if args.verify_only:
            command.append('--verify-only')
    report['command'] = list(map(str, command))
    # Source the provided DTK once; never install or change its packages.
    phase('load_bindings_then_infer' if not args.verify_only else 'local_cpu_bindings_only')
    with (output / 'runtime.log').open('wb') as log:
        child = subprocess.Popen(['/bin/bash', '-c', 'set -e; . /opt/dtk/env.sh >/dev/null; exec "$@"', 'preliminary', *command],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        last_heartbeat = 0
        with (output / 'runtime.log').open('r', encoding='utf-8', errors='replace') as events:
            while True:
                for line in events.read().splitlines():
                    print(line, flush=True)
                code = child.poll()
                if code is not None:
                    for line in events.read().splitlines():
                        print(line, flush=True)
                    break
                elapsed = time.monotonic() - started
                if time.monotonic() >= deadline:
                    raise TimeoutError('1740-second deadline before result packaging')
                if elapsed - last_heartbeat >= 15:
                    last_heartbeat = elapsed
                    logs = sorted(output.glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
                    active = logs[0] if logs else output / 'runtime.log'
                    print('REPRO_HEARTBEAT elapsed_seconds=' + str(round(elapsed, 1)) + ' latest_log=' + active.name + ' bytes=' + str(active.stat().st_size), flush=True)
                time.sleep(1)
    report['supervisor_exit_code'] = code
    if code:
        for name in ('local_check.json', 'public_cpu_check.json', 'rocblas_probe.json', 'preflight.json', 'vae_silu_preflight.json', 'mha_preflight.json', 'public_fl2va_report.json'):
            path = output / name
            if path.is_file():
                detail = json.loads(path.read_text())
                if detail.get('traceback'):
                    print('REPRO_FAILURE_DETAIL', name, detail.get('status'), flush=True)
                    print(detail['traceback'], flush=True)
    model_report = output / 'public_fl2va_report.json'
    if source_mode:
        reports = list((output / 'source_run').rglob('entry_report.json'))
        if len(reports) != 1:
            raise RuntimeError('Missing unique source-built entry report')
        model_report = reports[0]
        shutil.copyfile(model_report, output / 'source_entry_report.json')
    if model_report.exists():
        model = json.loads(model_report.read_text())
        report['public_fl2va_status'] = model['status']
        report['actual_model_executed'] = model.get('actual_model_executed', False)
        report['actual_h3_calls'] = model.get('actual_h3_calls')
        report['model_wall_seconds'] = model.get('model_wall_seconds')
        if source_mode:
            report['source_entry_wall_seconds'] = model.get('entry_wall_seconds')
    expected_status = 'LOCAL_PUBLIC_FL2VA_CPU_VERIFIED_GPU_PENDING' if args.verify_only else 'PUBLIC_FIRST_FRAME_SEVEN_NINE_MHA_GENERATION_VERIFIED_COMPLIANCE_PENDING'
    if source_mode:
        expected_status = 'SOURCE_BINDINGS_CPU_VERIFIED_GPU_PENDING' if args.verify_only else 'SOURCE_BUILT_PUBLIC_GENERATION_VERIFIED'
    if code != 0 or report.get('public_fl2va_status') != expected_status:
        raise RuntimeError('Runtime failed; return the complete result ZIP and console log')
    report['status'] = 'CPU_VERIFIED_GPU_PENDING' if args.verify_only else 'FULL_FL2VA_COMPLETED_COMPLIANCE_PENDING'
    if args.reference_video and not args.verify_only:
        phase('official_fl2va_comparison')
        candidate = next((output / 'source_run').rglob('public_dog.mp4')) if source_mode else output / 'public_dog.mp4'
        comparator = work / 'src/InfiniLM/examples/minimax_h3_compare.py'
        if digest(comparator) != 'e2c1dea3bc9f0c91e729d662f4dd7d5dcf3aebdd78cc02f9cbb61b816f425461':
            raise RuntimeError('Official comparator differs from the submitted source snapshot')
        report['official_comparator_sha256'] = digest(comparator)
        compare_env = dict(env, PYTHONPATH=str(root / 'site'))
        code = wait_process([sys.executable, comparator, '--tasks', 'fl2va', '--reference-fl2va-video', args.reference_video, '--infinilm-fl2va-video', candidate, '--model-path', args.model_path, '--output-dir', output / 'official_comparison'], compare_env, 'official_compare.log', max(1, deadline - time.monotonic() - 20))
        report['official_comparison_exit_code'] = code
        if code:
            raise RuntimeError('Official comparison failed; return result ZIP')
        report['official_comparison_status'] = 'COMPLETED'
        report['status'] = 'SOURCE_BUILD_FL2VA_AND_OFFICIAL_COMPARISON_COMPLETED_COMPLIANCE_PENDING'
    else:
        report['official_comparison_status'] = 'NOT_RUN_NO_REFERENCE_VIDEO' if not args.verify_only else 'NOT_RUN_CPU_ONLY'
    exit_code = 0
except Checkpoint as notice:
    report.update(status='SOURCE_BUILD_COMPLETE_INFERENCE_PENDING' if report.get('source_build_complete') else 'SOURCE_BUILD_CHECKPOINT_SAVED', checkpoint=str(notice))
    print('REPRO_CHECKPOINT', str(notice), flush=True)
    exit_code = 75
except BaseException:
    report.update(status='FAILED', traceback=traceback.format_exc())
    print(report['traceback'], flush=True)
finally:
    stop_tree()
    if report.get('source_build_workdir'):
        build_work = Path(report['source_build_workdir'])
        if (build_work / 'build_report.json').is_file():
            shutil.copyfile(build_work / 'build_report.json', output / 'build_report.json')
        if (build_work / 'logs').is_dir():
            shutil.copytree(build_work / 'logs', output / 'source_build_logs', dirs_exist_ok=True)
    report['elapsed_before_packaging_seconds'] = time.monotonic() - started
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    result = output.with_suffix('.zip')
    try:
        with zipfile.ZipFile(result, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file() and 'runtime_scratch' not in p.relative_to(output).parts:
                    z.write(p, p.relative_to(output).as_posix())
        print('REPRO_RESULT_ZIP', result, flush=True)
        print('REPRO_RESULT_SHA256', digest(result), flush=True)
    except BaseException:
        print('REPRO_PACKAGING_FAILED', traceback.format_exc(), flush=True)
        print('REPRO_RETURN_COMPLETE_DIRECTORY', output, flush=True)
        exit_code = 2
    if scratch is not None and (scratch / 'owned_preliminary').is_file() and (scratch / 'owned_preliminary').read_text() == run_id:
        try:
            shutil.rmtree(scratch)
        except OSError as error:
            print('REPRO_SCRATCH_CLEANUP_PENDING', scratch, str(error), flush=True)
    print('REPRO_STATUS', report['status'], flush=True)
    print('REPRO_TOTAL_SECONDS', round(time.monotonic() - started, 3), flush=True)
sys.exit(exit_code)
