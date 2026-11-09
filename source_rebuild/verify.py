"""Verify submitted source hashes and recursively reject bundled executables."""
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import zipfile

ROOT=Path(__file__).resolve().parent
FORBIDDEN={'.so','.a','.o','.hsaco','.co','.whl','.dll','.exe','.pyc','.pyo','.safetensors','.pt','.pth'}


def inspect(name,raw):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:raise RuntimeError('Unsafe source path: '+name)
    if path.suffix.lower() in FORBIDDEN or raw.startswith((b'\x7fELF',b'MZ')):
        raise RuntimeError('Binary/dependency artifact is not source: '+name)
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if archive.testzip() is not None:raise RuntimeError('Corrupt source archive')
            for child in archive.namelist():inspect(name+'/'+child,archive.read(child))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();manifest=json.loads((ROOT/'submission_manifest.json').read_text(encoding='utf-8'))
    expected={item['path'] for item in manifest['files']}|{'submission_manifest.json'}
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
            and not any(v in ('.git','work','__pycache__') for v in p.relative_to(ROOT).parts)}
    if actual!=expected:raise RuntimeError('Unexpected/missing submission files: '+repr(sorted(actual^expected)))
    for item in manifest['files']:
        path=ROOT/item['path'];raw=path.read_bytes();inspect(item['path'],raw)
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise RuntimeError('Changed source file: '+item['path'])
    with zipfile.ZipFile(ROOT/'source/native_sources.zip') as archive:
        sources=json.loads(archive.read('source_manifest.json'))
        for item in sources['files']:
            if hashlib.sha256(archive.read(item['path'])).hexdigest()!=item['sha256']:raise RuntimeError('Changed framework source')
    result=dict(status='SOURCE_ONLY_PACKAGE_VERIFIED',source_files=len(manifest['files']),native_source_files=len(sources['files']),bundled_precompiled_artifacts=0,bundled_dependency_wheels=0,model_executed=False)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
