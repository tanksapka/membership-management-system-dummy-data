"""
This module processes the source data and returns it in a normalized format.
"""
import os
import logging
import pandas as pd
import toml
from data_generator.web_data import FamilyNameScraper
from logging import config
from typing import Dict, Union, NoReturn


log_cfg = toml.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pyproject.toml'))
config.dictConfig(log_cfg)
_logger = logging.getLogger(__name__)


def get_src_path(src_file: str) -> Union[str, NoReturn]:
    """
    Creates full path to 'resources' folder for the provided scr_file and checks whether the file exists there.
    """
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', src_file)
    if not os.path.exists(src_path):
        raise FileNotFoundError('Source file not found: %s' % src_file)
    return src_path


def load_email_service_providers() -> pd.DataFrame:
    """
    Function to load e-mail service provider domains.
    """
    _logger.info('Loading e-mail service provider domains')
    src_path = get_src_path('email_service_providers.csv')

    df = pd.read_csv(src_path)
    _logger.info('Loading finished, %d rows loaded' % df.shape[0])
    return df


def load_mobile_phone_area_codes() -> pd.DataFrame:
    """
    Function to load Hungarian mobile phone area codes.
    """
    _logger.info('Loading mobile phone area codes')
    src_path = get_src_path('korzetszamok_mobil.xlsx')

    df: pd.DataFrame
    df = pd.read_excel(src_path)
    df.rename(columns={
        'Körzetszám': 'area_code',
        'Szolgáltató': 'service_provider',
        'Súly': 'weight',
    }, inplace=True)

    _logger.info('Loading finished, %d rows loaded' % df.shape[0])
    return df


def load_landline_phone_area_codes() -> pd.DataFrame:
    """
    Function to load Hungarian landline phone area codes.
    """
    _logger.info('Loading landline phone area codes')
    src_path = get_src_path('korzetszamok_vezetekes.xlsx')

    df: pd.DataFrame
    df = pd.read_excel(src_path, converters={'Irányítószám': int})
    df.rename(columns={
        'Település': 'settlement',
        'Megye': 'county',
        'Kistérség': 'micro_region',
        'Körzetszám': 'area_code',
        'Irányítószám': 'zip_code',
    }, inplace=True)

    _logger.info('Loading finished, %d rows loaded' % df.shape[0])
    return df


def load_messaging_platform_data() -> pd.DataFrame:
    """
    Function to load message sending platform data (and their availability for phone and email).
    """
    _logger.info('Loading messaging platform data')
    src_path = get_src_path('uzenetkuldo_appok.xlsx')

    df: pd.DataFrame
    df = pd.read_excel(src_path)

    _logger.info('Loading finished, %d rows loaded' % df.shape[0])
    return df


def load_settlement_data() -> pd.DataFrame:
    """
    Function to load settlement data on ZIP code level.
    """
    _logger.info('Loading settlement data')
    src_path = get_src_path('IrszHnk.csv')

    df: pd.DataFrame
    with open(src_path, 'r') as fh:
        df = pd.read_csv(fh, delimiter=';')
    df = df[['Helység.megnevezése', 'IRSZ', 'Településrész', 'Helység.KSH kódja', 'Helység.jogállása',
             'Megye megnevezése', 'Járáskódja', 'Járásneve', 'Lakó-népesség']]
    df.rename(columns={
        'Helység.megnevezése': 'settlement_name',
        'IRSZ': 'zip_code',
        'Településrész': 'part_of_settlement',
        'Helység.KSH kódja': 'settlement_ksh_code',
        'Helység.jogállása': 'settlement_legal_status',
        'Megye megnevezése': 'county_name',
        'Járáskódja': 'district_code',
        'Járásneve': 'district_name',
        'Lakó-népesség': 'resident_population',
    }, inplace=True)
    df.settlement_name = df.settlement_name.apply(lambda x: x.replace('õ', 'ő'))
    df.settlement_name = df.settlement_name.apply(lambda x: x.replace('û', 'ű'))

    _logger.info('Loading finished, %d rows loaded' % df.shape[0])
    return df


def load_street_names() -> pd.DataFrame:
    """
    Function to load street name data.
    """
    _logger.info('Loading street name data')
    src_path = get_src_path('utcanevek.csv')
    required_spaces = [
        'fasor', 'körönd', 'körtér', 'körút', 'körútja', 'korzó', 'köz', 'lakópark', 'lakótelep', 'liget', 'major',
        'park', 'parkja', 'sétány', 'sor', 'sugárút', 'telep', 'tér', 'tere', 'udvar', 'udvara', 'út', 'utca', 'utcája',
        'útja', 'villasor',
    ]

    df: pd.DataFrame
    with open(src_path, 'r', encoding='utf-8') as fh:
        df = pd.read_csv(fh, delimiter=',')
    df.rename(columns={'Név': 'street_name', 'Útszakasz': 'street_count'}, inplace=True)
    df = df.loc[(df['street_count'] >= 5) & ~(df['street_name'].apply(lambda x: x[0].isnumeric()))]
    df.insert(len(df.columns), 'str_ending', df['street_name'].apply(lambda x: x[x.rfind(' '):].strip().lower()))
    df = df.loc[df['str_ending'].isin(required_spaces)]
    df.drop(columns=['str_ending'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    _logger.info('Loading finished, %d rows loaded' % df.shape[0])
    return df


def load_first_names() -> pd.DataFrame:
    """
    Function to load approved Hungarian first names.
    """
    _logger.info('Loading first names')
    src_files = {
        'male': 'osszesffi.txt',
        'female': 'osszesnoi.txt',
    }
    df_dict: Dict[str, pd.DataFrame]
    df_dict = dict()

    for gender, file in src_files.items():
        src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', file)
        if not os.path.exists(src_path):
            raise FileNotFoundError('Source file not found: %s' % src_path)
        with open(src_path, 'r') as fh:
            df_dict[gender] = pd.read_csv(fh)
            df_dict[gender].columns = ['first_name']
            df_dict[gender].insert(1, 'gender', gender)

    df = pd.concat(df_dict.values())
    _logger.info('Loading finished, %d rows loaded' % df.shape[0])
    return df


def _load_most_common_last_names() -> pd.DataFrame:
    """
    Function to load a merged list of the top 100 most common family names in Hungary.
    """
    _logger.info('Loading most common last names')
    files = list()
    dir_scan = os.walk(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources'))
    for dirpath, dirnames, filenames in dir_scan:
        files = [file for file in filenames if file.startswith('kozerdeku_csaladnev_')]

    df_dict: Dict[str, pd.DataFrame]
    df_dict = dict()

    col_name_map = {
        'Családi név': 'last_name',
        'Születési családi név': 'last_name',
    }

    for file in files:
        src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', file)
        df_dict[file] = pd.read_excel(src_path)
        df_dict[file].rename(columns=col_name_map, inplace=True)
        df_dict[file] = df_dict[file][['last_name']]
        df_dict[file]['last_name'] = df_dict[file]['last_name'].apply(lambda x: x.capitalize())

    df = pd.concat(df_dict.values())
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)

    _logger.info('Loading finished, %d rows loaded' % df.shape[0])
    return df


def _load_enriched_last_names() -> pd.DataFrame:
    """
    Function to load an enriched list of family names occurring in Hungary.
    """
    _logger.info('Loading enriched last names')
    src_file = 'csaladnevek_listaja.txt'
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', src_file)

    df: pd.DataFrame
    if os.path.exists(src_path):
        with open(src_path, 'r', encoding='utf-8') as fh:
            df = pd.read_csv(fh, header=None, names=['last_name'])
    else:
        _logger.warning('Source file not found: %s' % src_file)
        _logger.info('Attempt to scrape data from source webpage')
        scraper = FamilyNameScraper()
        results = scraper.execute(src_path)
        df = pd.DataFrame(data=pd.Series(data=results, name='last_name'))
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)

    _logger.info('Loading finished, %d rows loaded' % df.shape[0])
    return df


def load_last_names() -> pd.DataFrame:
    """
    Function to load Hungarian family names flagging the top 100 (or so) most common ones.
    """
    _logger.info('Loading last names')
    common_names = _load_most_common_last_names()
    df = pd.merge(_load_enriched_last_names(), common_names, how='outer', on='last_name')
    df.insert(1, 'most_common_flag', 'N')
    df.most_common_flag = df.last_name.apply(lambda x: 'Y' if x in common_names.last_name.values else 'N')
    _logger.info('Loading finished, %d rows loaded' % df.shape[0])
    return df


def load_organization_stats() -> pd.DataFrame:
    """
    Function to load NGO organization units and its member statistics.
    """
    _logger.info('Loading organization member statistics')
    src_path = get_src_path('alapszervezeti_letszamok.xlsx')

    df: pd.DataFrame
    df = pd.read_excel(src_path)
    df.rename(columns={
        'Alapszervezet': 'organization',
        'Tag Létszám 2018': 'members_2018',
        '2019 taglétszám': 'members_2019',
        '2020 taglétszám': 'members_2020',
        '2021 taglétszám': 'members_2021',
    }, inplace=True)

    _logger.info('Loading finished, %d rows loaded' % df.shape[0])
    return df
