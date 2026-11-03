"""Export/import one completed problem with hashes, protocol and code checks."""
import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'pca_collocation/configs/benchmark_full.json'

# Reviewed compatibility only: numerical training is unchanged by the atomic-I/O fix.
# Exact source hashes keep this exception from accepting arbitrary future edits.
IO_FIX_CODE_GROUP=frozenset((
    '3bac3b4d024e4d3894c518731fa18930b06920066b57c4d0beeb7408f9a43337',
    'ff405605bcebff5046ba1eaef37cd75db20aef0d1ee9e0f66a44c0fd837ce747',
))


def compatible_code(previous,current):
    return previous==current or {previous,current}<=IO_FIX_CODE_GROUP


def compatible_environment(previous,current):
    for key in ('python','platform','packages'):
        if previous.get(key)!=current.get(key):raise ValueError(f'Environment changed: {key}; use the original environment')
    revisions=set(previous.get('execution_code_revisions',[]))|{previous['code_sha256']}
    if not all(compatible_code(revision,current['code_sha256']) for revision in revisions):
        raise ValueError('Unrecognized code change; use the original checkout or a new run_id')
    # Preserve the original provenance, recording reviewed compatible resumes separately.
    return dict(previous,execution_code_revisions=sorted(revisions|{current['code_sha256']}))


def digest(data):return hashlib.sha256(data).hexdigest()


def code_digest():
    files=sorted((ROOT/'src/pinn_pca').glob('*.py'))+[ROOT/'scripts/run_collocation_benchmark.py']
    # Ignore checkout line endings (Windows CRLF versus Linux LF).
    return digest(b''.join(p.relative_to(ROOT).as_posix().encode()+b'\0'+p.read_text(encoding='utf-8-sig').encode('utf-8')+b'\0' for p in files))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    export=commands.add_parser('export')
    export.add_argument('--problem',choices=('burgers','wave','allen_cahn'),required=True)
    export.add_argument('--output',type=Path,help='Default: benchmark_artifacts/<run_id>_<problem>.zip')
    imp=commands.add_parser('import')
    imp.add_argument('archive',type=Path)
    args=parser.parse_args()
    c=json.loads(CONFIG.read_text());out=ROOT/'pca_collocation/results/benchmark'/c['run_id']
    protocol=digest(json.dumps(c,sort_keys=True).encode())
    if args.action=='export':
        problem=args.problem
        stored=json.loads((out/'config.json').read_text())
        if stored!=c:raise ValueError('Stored protocol differs from current configuration')
        report=json.loads((out/problem/'report_complete.json').read_text())
        assert report['passed'] and report['completed_runs']==50
        markers=list((out/problem).glob('*/seed_*/completed.json'));assert len(markers)==50
        assert all(json.loads(p.read_text())['passed'] for p in markers)
        files=[out/'config.json',out/'references'/f'{problem}.npz',out/'references'/f'{problem}_audit.json']
        files+=sorted(p for p in (out/problem).rglob('*') if p.is_file() and p.suffix not in ('.tmp','.log') and p.name!='last_error.txt')
        provenance=out/problem/'environment.json'
        if not provenance.exists():raise ValueError('Missing environment.json: use the current runner for the study')
        environment=json.loads(provenance.read_text())
        revisions=set(environment.get('execution_code_revisions',[]))|{environment['code_sha256']}
        if not all(compatible_code(r,code_digest()) for r in revisions):raise ValueError('Code changed since training; export from the original or reviewed compatible checkout')
        manifest=dict(format=1,problem=problem,run_id=c['run_id'],protocol_sha256=protocol,
                      code_sha256=code_digest(),training_code_revisions=sorted(revisions),
                      files={p.relative_to(out).as_posix():digest(p.read_bytes()) for p in files})
        target=args.output or ROOT/'benchmark_artifacts'/f'{c["run_id"]}_{problem}.zip'
        if target.exists():raise FileExistsError(f'Archive exists: {target}; use a different --output')
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.tmp')
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            archive.writestr('archive_manifest.json',json.dumps(manifest,indent=2))
            for p in files:archive.write(p,p.relative_to(out).as_posix())
        temporary.replace(target)
        print(f'Exported {problem}: {target} ({target.stat().st_size/1024**2:.1f} MiB)')
    else:
        with zipfile.ZipFile(args.archive) as archive:
            manifest=json.loads(archive.read('archive_manifest.json'))
            assert manifest['format']==1 and manifest['run_id']==c['run_id']
            assert manifest['protocol_sha256']==protocol,'Protocol mismatch'
            assert compatible_code(manifest['code_sha256'],code_digest()),'Use the same or reviewed compatible code revision as the training machines'
            assert all(compatible_code(r,code_digest()) for r in manifest.get('training_code_revisions',[manifest['code_sha256']])), 'Unrecognized training revision'
            problem=manifest['problem'];assert problem in c['problems']
            names=archive.namelist();assert len(names)==len(set(names))
            assert set(names)==set(manifest['files'])|{'archive_manifest.json'}
            # Validate the entire archive and all existing-file conflicts before writing anything.
            for name,sha in manifest['files'].items():
                p=PurePosixPath(name)
                if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe archive path')
                allowed=name=='config.json' or p.parts[0]==problem or name in (f'references/{problem}.npz',f'references/{problem}_audit.json')
                if not allowed:raise ValueError(f'Unexpected archive member: {name}')
                if stat.S_ISLNK(archive.getinfo(name).external_attr>>16):raise ValueError('Symbolic links are not allowed')
                target=out.joinpath(*p.parts)
                if not target.resolve().is_relative_to(out.resolve()):raise ValueError('Path escapes output directory')
                data=archive.read(name)
                if digest(data)!=sha:raise ValueError(f'Checksum failed: {name}')
                if target.exists() and digest(target.read_bytes())!=sha:
                    if name!='config.json' or json.loads(target.read_text())!=c:
                        raise FileExistsError(f'Conflicting local result: {target}')
            if json.loads(archive.read('config.json'))!=c:raise ValueError('Invalid stored configuration')
            report=json.loads(archive.read(f'{problem}/report_complete.json'))
            assert report['passed'] and report['completed_runs']==50
            for name in manifest['files']:
                target=out.joinpath(*PurePosixPath(name).parts)
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    temporary=target.with_suffix(target.suffix+'.tmp');temporary.write_bytes(archive.read(name));temporary.replace(target)
        print(f'Imported {problem} into {out}; existing identical files preserved. Run --aggregate after importing all three problems.')


if __name__=='__main__':main()
