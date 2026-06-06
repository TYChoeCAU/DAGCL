import argparse
import sys
import ast
from pathlib import Path

from recbole_gnn.quick_start import run_recbole_gnn

PROJECT_ROOT = Path(__file__).resolve().parent


def _coerce_cli_value(value: str):
    """Best-effort conversion for RecBole config overrides.

    Supports both normal RecBole-style values and our aux parameters, e.g.
    --tail_weight=0.1, --aux_modules tail,head, --topk "[20,50]".
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    low = s.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    # ast.literal_eval cannot parse bare strings like tail,head or auxhard.
    try:
        if any(c in s for c in [".", "e", "E"]):
            return float(s)
        return int(s)
    except Exception:
        return s


def _parse_unknown_args(unknown):
    """Parse unknown CLI args into a config_dict for RecBole Config.

    The original script ignored these args after parse_known_args(), so custom
    aux options such as --aux_modules or --tail_weight could silently disappear.
    This parser accepts both '--key=value' and '--key value'.
    """
    cfg = {}
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            i += 1
            continue
        tok = tok[2:]
        if not tok:
            i += 1
            continue
        if "=" in tok:
            key, value = tok.split("=", 1)
            cfg[key] = _coerce_cli_value(value)
            i += 1
            continue
        key = tok
        if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            cfg[key] = _coerce_cli_value(unknown[i + 1])
            i += 2
        else:
            cfg[key] = True
            i += 1
    return cfg


def _is_inside_project(path: Path):
    try:
        path.resolve().relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def _sanitize_project_paths(config_dict):
    """Keep CLI path overrides inside this project.

    This prevents accidentally reusing data/n2v files from sibling experiment
    folders when an old command is pasted.
    """
    if not config_dict:
        return config_dict

    data_path = config_dict.get('data_path')
    if data_path:
        data_path = Path(str(data_path)).expanduser()
        resolved = data_path if data_path.is_absolute() else PROJECT_ROOT / data_path
        if not _is_inside_project(resolved):
            local_data = PROJECT_ROOT / 'dataset'
            print(f'[run_recbole_gnn] replacing external data_path with {local_data}')
            config_dict['data_path'] = str(local_data)

    n2v_path = config_dict.get('n2v_path')
    if n2v_path:
        n2v_path = Path(str(n2v_path)).expanduser()
        resolved = n2v_path if n2v_path.is_absolute() else PROJECT_ROOT / n2v_path
        if not _is_inside_project(resolved):
            local_n2v = PROJECT_ROOT / 'n2v' / n2v_path.name
            if local_n2v.exists():
                print(f'[run_recbole_gnn] replacing external n2v_path with {local_n2v}')
                config_dict['n2v_path'] = str(local_n2v)
            else:
                raise ValueError(
                    f'External n2v_path is not allowed: {n2v_path}. '
                    f'Copy it into {PROJECT_ROOT / "n2v"} first.'
                )

    return config_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, unknown = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    cli_config_dict = _parse_unknown_args(unknown)
    cli_config_dict = _sanitize_project_paths(cli_config_dict)

    if cli_config_dict:
        print('[run_recbole_gnn] parsed CLI config overrides:')
        for key in sorted(cli_config_dict):
            print(f'  {key}: {cli_config_dict[key]}')

    # RecBole's Config also inspects sys.argv.  If we leave custom
    # arguments such as `--tail_topk 20` in sys.argv, RecBole may parse
    # the value token (`20`) as a separate boolean key, producing logs like
    # `20 = True`.  We already converted all unknown CLI arguments into
    # config_dict above, so clear sys.argv before constructing Config.
    sys.argv = [sys.argv[0]]

    run_recbole_gnn(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=cli_config_dict if cli_config_dict else None,
    )
