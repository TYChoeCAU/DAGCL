import logging
import hashlib
import os
import uuid
from datetime import datetime
from logging import getLogger
from pathlib import Path
from recbole.utils import init_seed, set_color
from recbole.utils.logger import RemoveColorFilter, ensure_dir, log_colors_config

from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def init_unique_logger(config):
    """Initialize a fresh log file for every process run."""
    log_root = './log/'
    model_dir = os.path.join(log_root, config['model'])
    ensure_dir(model_dir)

    config_str = ''.join([str(value) for value in config.final_config_dict.values()])
    md5 = hashlib.md5(config_str.encode(encoding='utf-8')).hexdigest()[:6]
    run_time = datetime.now().strftime('%b-%d-%Y_%H-%M-%S-%f')
    run_id = f'{os.getpid()}-{uuid.uuid4().hex[:8]}'
    logfilename = f"{config['model']}-{config['dataset']}-{run_time}-{md5}-{run_id}.log"
    logfilepath = os.path.join(model_dir, logfilename)

    fileformatter = logging.Formatter(
        '%(asctime)-15s %(levelname)s  %(message)s',
        '%a %d %b %Y %H:%M:%S'
    )
    try:
        import colorlog
        streamformatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)-15s %(levelname)s  %(message)s',
            '%d %b %H:%M',
            log_colors=log_colors_config
        )
    except ImportError:
        streamformatter = logging.Formatter(
            '%(asctime)-15s %(levelname)s  %(message)s',
            '%d %b %H:%M'
        )

    state = config['state']
    if state is None or state.lower() == 'info':
        level = logging.INFO
    elif state.lower() == 'debug':
        level = logging.DEBUG
    elif state.lower() == 'error':
        level = logging.ERROR
    elif state.lower() == 'warning':
        level = logging.WARNING
    elif state.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    file_handler = logging.FileHandler(logfilepath, mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(fileformatter)
    file_handler.addFilter(RemoveColorFilter())

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(streamformatter)

    logging.basicConfig(level=level, handlers=[stream_handler, file_handler], force=True)
    return logfilepath


def _is_inside_project(path):
    try:
        Path(path).resolve().relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def keep_paths_inside_project(config):
    """Rewrite accidental sibling-project paths to local project paths."""
    data_path = config['data_path']
    if data_path:
        data_path = Path(str(data_path)).expanduser()
        resolved = data_path if data_path.is_absolute() else PROJECT_ROOT / data_path
        if not _is_inside_project(resolved):
            local_data = PROJECT_ROOT / 'dataset'
            config['data_path'] = str(local_data)

    n2v_path = config['n2v_path']
    if n2v_path:
        n2v_path = Path(str(n2v_path)).expanduser()
        resolved = n2v_path if n2v_path.is_absolute() else PROJECT_ROOT / n2v_path
        if not _is_inside_project(resolved):
            local_n2v = PROJECT_ROOT / 'n2v' / n2v_path.name
            if local_n2v.exists():
                config['n2v_path'] = str(local_n2v)
            else:
                raise ValueError(
                    f'External n2v_path is not allowed: {n2v_path}. '
                    f'Copy it into {PROJECT_ROOT / "n2v"} first.'
                )

    return config


def run_recbole_gnn(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset
    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    config = keep_paths_inside_project(config)
    try:
        assert config["enable_sparse"] in [True, False, None]
    except AssertionError:
        raise ValueError("Your config `enable_sparse` must be `True` or `False` or `None`")
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    logfilepath = init_unique_logger(config)
    logger = getLogger()

    logger.info(set_color('log file', 'green') + f': {logfilepath}')
    logger.info(set_color('project root', 'green') + f': {PROJECT_ROOT}')
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    config = keep_paths_inside_project(config)
    try:
        assert config["enable_sparse"] in [True, False, None]
    except AssertionError:
        raise ValueError("Your config `enable_sparse` must be `True` or `False` or `None`")
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'model': config['model'],
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
