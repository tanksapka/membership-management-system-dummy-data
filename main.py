import argparse
import logging
import toml
from data_generator.data_generator import DummyDataGenerator, pd
from data_generator.id_utils import map_id_fields
from logging import config
from pathlib import Path


log_cfg = toml.load(Path(__file__).parent.joinpath('pyproject.toml'))
config.dictConfig(log_cfg)
_logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dummy data creation and export results to output folder")
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='output folder path')
    parser.add_argument('-t', '--file_type', type=str, choices=['csv', 'xls'], default='csv', help='output file type')
    parser.add_argument('-i', '--id_type', type=str, choices=['int', 'uuid'], default='int', help='type of id fields')
    return parser


def run_data_generation(output_folder: str, file_format: str, id_type: str) -> None:
    _logger.info(f'Received inputs: {output_folder} and {file_format}')
    path = Path(output_folder)
    if not path.exists():
        _logger.info(f'Creating output folder ({path}) as it does not exist')
        path.mkdir(parents=True, exist_ok=True)
    ddg = DummyDataGenerator()
    dummy_data = ddg.generate_dummy_data()

    if id_type == 'uuid':
        dummy_data = map_id_fields(dummy_data)

    if file_format == 'csv':
        for dst, df in dummy_data.items():
            _logger.info(f'Saving {dst} data to csv')
            df.to_csv(path.resolve().joinpath(f'{dst}.csv'), sep='\t', index_label='id')

    if file_format == 'xls':
        with pd.ExcelWriter(path.resolve().joinpath('output.xlsx'), mode='w') as writer:
            for dst, df in dummy_data.items():
                _logger.info(f'Saving {dst} data to Excel')
                df.to_excel(writer, sheet_name=dst, index_label='id')


if __name__ == '__main__':
    args = create_parser()
    values = args.parse_args()
    run_data_generation(values.output_folder, values.file_type, values.id_type)
