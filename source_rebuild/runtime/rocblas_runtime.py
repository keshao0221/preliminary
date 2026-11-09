"""Select an existing gfx936 catalog from the active DTK installation only."""
import hashlib
import os
from pathlib import Path

VARIABLE = 'ROCBLAS_TENSILE_LIBPATH'


class CatalogUnavailable(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__('No usable gfx936 catalog in active DTK: ' + str(report['attempts']))


def inspect_catalog(path):
    path = Path(path).resolve()
    catalog = path / 'TensileLibrary_lazy_gfx936.dat'
    if not catalog.is_file():
        catalog = path / 'TensileLibrary.dat'
    objects = sorted(path.glob('*gfx936*.co')) + sorted(path.glob('*gfx936*.hsaco'))
    if not catalog.is_file() or catalog.stat().st_size == 0 or not objects:
        raise RuntimeError('Missing readable gfx936 catalog/code objects: ' + str(path))
    with catalog.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    # Test actual readability rather than just path existence.
    for obj in objects:
        if obj.stat().st_size == 0:
            raise RuntimeError('Empty gfx936 object: ' + str(obj))
        with obj.open('rb') as stream:
            stream.read(1)
    return dict(directory=str(path), catalog=catalog.name, catalog_bytes=catalog.stat().st_size,
                catalog_sha256=digest, gfx936_code_objects=len(objects))


def select_catalog(env, dtk=Path('/opt/dtk')):
    root = Path(dtk).resolve()
    inherited = env.get(VARIABLE)
    candidates = ([Path(inherited)] if inherited else []) + [
        root / relative for relative in ('lib/rocblas/library_gpu5', 'lib64/rocblas/library_gpu5',
        'rocblas/library_gpu5', 'lib/rocblas/library', 'lib64/rocblas/library', 'rocblas/library')]
    attempts = []
    seen = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            if not path.is_relative_to(root):
                raise RuntimeError('Outside active DTK installation')
            selected = inspect_catalog(path)
        except (OSError, RuntimeError) as error:
            attempts.append(dict(directory=str(path), usable=False, reason=str(error)))
            continue
        attempts.append(dict(directory=str(path), usable=True))
        return dict(status='GFX936_ROCBLAS_CATALOG_SELECTED_GPU_PENDING', dtk_root=str(root),
                    inherited_value=inherited, selected=selected, attempts=attempts,
                    environment_updates={VARIABLE: selected['directory']}, system_files_modified=False)
    raise CatalogUnavailable(dict(status='NO_USABLE_GFX936_ROCBLAS_CATALOG', dtk_root=str(root),
        inherited_value=inherited, attempts=attempts, system_files_modified=False))


def mapped_blas():
    names = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        path = line.split()[-1]
        if path.startswith('/') and any(part in Path(path).name for part in ('librocblas', 'libhipblas')):
            names.add(str(Path(path).resolve()))
    return sorted(names)


def model_binding(expected):
    actual = os.environ.get(VARIABLE)
    if actual != expected:
        raise RuntimeError('Model rocBLAS catalog override changed')
    return dict(catalog_directory=actual, mapped_blas=mapped_blas())
