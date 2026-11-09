"""Use the selected competition interpreter and toolchain without installing packages."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import sys


def check(root, role):
    # Works both as a CLI under tools/ and when imported by build.py/run.py.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from tools.platform_api import check as check_project_api
    return check_project_api(root,role)


def build_environment(work):
    import torch
    work=Path(work)
    env=os.environ.copy()
    env.update(INFINI_ROOT=str(work/'infini'),PYTHON=sys.executable,HYGON_ARCH='gfx936',PYTORCH_ROCM_ARCH='gfx936',
               INFINILM_CXX11_ABI='1',XMAKE_ROOT='y',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
               PIP_NO_INDEX='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env['PYTHONPATH']=os.pathsep.join(filter(None,[str(work/'src/InfiniCore/python'),str(work/'src/InfiniLM/python'),env.get('PYTHONPATH','')]))
    env['LD_LIBRARY_PATH']=os.pathsep.join(filter(None,[str(work/'infini/lib'),str(Path(torch.__file__).parent/'lib'),env.get('LD_LIBRARY_PATH','')]))
    # Preserve the platform's XMAKE_RCFILES and dependency/package cache paths.
    env['XMAKE_GLOBALDIR']=str(work/'xmake-global')
    env['XMAKE_TMPDIR']=str(work/'cache/tmp')
    for path in ('infini','xmake-global','cache/tmp'):(work/path).mkdir(parents=True,exist_ok=True)
    return env


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role',choices=('compile','runtime','metrics'),required=True)
    args=parser.parse_args()
    check(Path(__file__).resolve().parents[1],args.role)
