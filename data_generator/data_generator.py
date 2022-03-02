"""
This module creates the dummy data replicating the memberdb data model and exports it into csv files.
"""
import data_generator.source_data as sd
import datetime
import logging
import os
import pandas as pd
import random
import toml
from collections import defaultdict
from inspect import getmembers, isfunction
from logging import config
from string import ascii_uppercase, digits
from typing import Any, DefaultDict, Dict, List


log_cfg = toml.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pyproject.toml'))
config.dictConfig(log_cfg)
_logger = logging.getLogger(__name__)


def trunc_gauss(mu: float, sigma: float, bottom: float, top: float) -> float:
    a = random.gauss(mu, sigma)
    while not (bottom <= a <= top):
        a = random.gauss(mu, sigma)
    return a


class DummyDataGenerator:
    """
    Class to facilitate dummy data creation. This includes:
        - Person data (and corresponding contacts)
        - Organization data (and corresponding contacts)
        - Organization-person relational data
    """
    def __init__(self) -> None:
        _logger.info('Initializing DummyData class')

        self.loader_functions = getmembers(sd, isfunction)
        self.resources = {
            fn_name.replace('load_', ''): fn() for fn_name, fn in self.loader_functions if fn_name.startswith('load_')
        }
        self._load_config()

        self.person_data = DummyPersonData(self.resources, self.config)
        self.organization_data = DummyOrganizationData(self.resources, self.config)
        self.address_data = DummyAddressData(self.resources, self.config)
        self.phone_data = DummyPhoneData(self.resources, self.config)
        self.email_data = DummyEmailData(self.resources, self.config)
        self.membership_data = DummyMembershipData(self.resources, self.config)

    def _load_config(self) -> None:
        py_project = toml.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pyproject.toml'))
        self._config = py_project.get('dummy', dict()).get('config', dict())

    @property
    def config(self) -> Dict[str, Any]:
        return self._config.copy()

    def generate_dummy_data(self) -> Dict[str, pd.DataFrame]:
        """
        Main method to generate the dummy data for the database's data model. Returns dictionary of DataFrames.
        """
        df_ppl = self.person_data()
        df_ppl.index += 1
        df_org = self.organization_data(df_ppl)
        df_org.index += 1
        df_address = self.address_data(df_ppl, df_org)
        df_address.index += 1
        df_phone = self.phone_data(df_address)
        df_phone.index += 1
        df_email = self.email_data(df_ppl, df_org)
        df_email.index += 1
        df_membership = self.membership_data(df_ppl, df_org)
        df_membership.index += 1

        df_ppl.drop(columns=['organization'], inplace=True)

        current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        df_ppl.insert(len(df_ppl.columns), "created_on", current_timestamp)
        df_ppl.insert(len(df_ppl.columns), "created_by", os.getlogin())

        df_org.insert(len(df_org.columns), "created_on", current_timestamp)
        df_org.insert(len(df_org.columns), "created_by", os.getlogin())

        df_address.insert(len(df_address.columns), "created_on", current_timestamp)
        df_address.insert(len(df_address.columns), "created_by", os.getlogin())

        df_phone.insert(len(df_phone.columns), "created_on", current_timestamp)
        df_phone.insert(len(df_phone.columns), "created_by", os.getlogin())

        df_email.insert(len(df_email.columns), "created_on", current_timestamp)
        df_email.insert(len(df_email.columns), "created_by", os.getlogin())

        df_membership.insert(len(df_membership.columns), "created_on", current_timestamp)
        df_membership.insert(len(df_membership.columns), "created_by", os.getlogin())

        return {
            'person': df_ppl[['created_on', 'created_by']].copy(),
            'person_data': df_ppl,
            'organization': df_org[['created_on', 'created_by']].copy(),
            'organization_data': df_org,
            'address': df_address,
            'phone': df_phone,
            'email': df_email,
            'membership': df_membership,
        }


class DummyDataBase:
    """
    Base class to take care of source handover in init.
    """
    def __init__(self, src: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> None:
        _logger.info('Initializing %s class' % self.__class__.__name__)
        self.src = src
        self.cfg = cfg

        self._setup_member_distribution()

    def _setup_member_distribution(self) -> None:
        df = self.src.get('organization_stats', pd.DataFrame()).copy()
        df.insert(1, 'member_count', df.mean(axis=1).apply(round, args=0))
        df.drop(columns=[col for col in df.columns if col.startswith('members_20')], inplace=True)
        self._member_distribution = df

    @property
    def seed(self) -> int:
        return self.cfg.get('seed', 1)

    @property
    def base_date(self) -> datetime.date:
        return datetime.datetime.strptime(self.cfg.get('base_date', '2020-12-31'), '%Y-%m-%d').date()

    @property
    def member_distribution(self) -> pd.DataFrame:
        return self._member_distribution

    @property
    def messaging_platform_data(self) -> pd.DataFrame:
        return self.src.get('messaging_platform_data').copy()

    def __call__(self, *args, **kwargs):
        _logger.info('Generating data in %s class' % self.__class__.__name__)


class DummyPersonData(DummyDataBase):
    """
    Class to create dummy data for members of the NGO.
    """
    def __init__(self, src: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> None:
        super().__init__(src, cfg)
        self.df_person = pd.DataFrame()

    @property
    def gender_distribution(self) -> Dict[str, float]:
        return self.cfg.get('gender_distribution', dict()).copy()

    @property
    def gender_map(self) -> Dict[str, int]:
        return self.cfg.get('gender_id_map', dict()).copy()

    @property
    def first_names(self) -> pd.DataFrame:
        return self.src.get('first_names', pd.DataFrame())

    @property
    def last_names(self) -> pd.DataFrame:
        return self.src.get('last_names', pd.DataFrame())

    @property
    def age_distribution(self) -> Dict[str, int]:
        return self.cfg.get('age_distribution', dict())

    @property
    def membership_id_map(self) -> Dict[int, int]:
        return {int(k): v for k, v in self.cfg.get('membership_id_map', dict()).items()}

    def generate_first_names(self) -> pd.DataFrame:
        """
        Generates a dataframe of first names based on given gender and organization distribution.
        """
        first_name_dfs = list()
        for idx, row in self.member_distribution.iterrows():
            total_cnt = row.at['member_count']
            gender_map = {k: round(v * total_cnt) for k, v in self.gender_distribution.items()}
            if total_cnt != sum(gender_map.values()):
                gender_map['male'] = gender_map['male'] + total_cnt - sum(gender_map.values())

            for gender, count in gender_map.items():
                if gender in ('male', 'female'):
                    tmp_df = self.first_names.loc[self.first_names.gender == gender].sample(
                        n=count, random_state=self.seed
                    )
                else:
                    tmp_df = self.first_names.sample(n=count, random_state=self.seed)
                    tmp_df['gender'] = 'other'
                tmp_df.insert(0, 'organization', row.at['organization'])
                tmp_df.insert(len(tmp_df.columns), 'gender_id', self.gender_map[gender])
                first_name_dfs.append(tmp_df.copy())

        df_first_names = pd.concat(first_name_dfs)
        df_first_names.reset_index(drop=True, inplace=True)
        return df_first_names.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def generate_last_names(self, count: int) -> pd.DataFrame:
        """
        Generates a dataframe of last names.
        """
        return self.last_names.sample(n=count, random_state=self.seed).reset_index(drop=True).iloc[:, :1]

    def generate_date_of_birth(self, count: int) -> pd.DataFrame:
        """
        Generates a dataframe of date of birth values (already converted to database compatible string).
        """
        random.seed = self.seed
        days = [
            int(trunc_gauss(**{k: v * 365 for k, v in self.age_distribution.items()}))
            for _ in range(count)
        ]
        return pd.DataFrame(
            [(self.base_date - datetime.timedelta(days=day)).strftime('%Y-%m-%d') for day in days],
            columns=['birthdate']
        )

    def generate_mother_name(self, count: int) -> pd.DataFrame:
        """
        Generates a dataframe of mother's names.
        """
        df = pd.merge(
            left=(
                self.first_names
                    .loc[self.first_names.gender == 'female']
                    .sample(n=count, replace=True, random_state=self.seed)
                    .reset_index(drop=True)
                    .iloc[:, :1]
            ),
            right=self.last_names.sample(n=count, random_state=self.seed).reset_index(drop=True).iloc[:, :1],
            left_index=True,
            right_index=True,
        )
        df['mother_name'] = df['last_name'] + ' ' + df['first_name']
        return df.iloc[:, -1:]

    def generate_identity_card_number(self, count: int) -> pd.DataFrame:
        """
        Generates a dataframe of id card numbers
        """
        random.seed = self.seed
        return pd.DataFrame(
            data=[
                ''.join(random.choices(digits, k=6)) + ''.join(random.choices(ascii_uppercase, k=2))
                for _ in range(count)
            ],
            columns=['identity_card_number'],
        )

    def map_membership_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds membership category codes based on age.
        """
        df.insert(len(df.columns), 'membership_fee_category_id', self.membership_id_map.get(-1))
        for age, id_ in self.membership_id_map.items():
            mask = (df.birthdate
                      .apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
                      .apply(lambda x: (self.base_date - x.date()).days) <= age * 365)
            df.loc[mask, 'membership_fee_category_id'] = id_
        return df

    def __call__(self) -> pd.DataFrame:
        """
        Runs data generation for person data. Output dataframe will also contain the organization where the given member
        belongs to.
        """
        super().__call__()
        df = self.generate_first_names()
        df = pd.merge(df, self.generate_last_names(len(df.index)), left_index=True, right_index=True)
        df.insert(len(df.columns), 'name', df['last_name'] + ' ' + df['first_name'])

        df = pd.merge(df, self.generate_date_of_birth(len(df.index)), left_index=True, right_index=True)
        df = pd.merge(df, self.generate_mother_name(len(df.index)), left_index=True, right_index=True)
        df = pd.merge(df, self.generate_identity_card_number(len(df.index)), left_index=True, right_index=True)
        self.map_membership_category(df)

        df.insert(0, 'person_id', [x for x in range(1, len(df.index) + 1)])
        df.insert(1, 'registration_number', df['person_id'])
        df.insert(2, 'membership_id', df['person_id'].apply(lambda x: 'A{:9>6}'.format(x)))
        df.insert(len(df.columns), 'notes', None)
        df.insert(len(df.columns), 'valid_from', self.base_date.strftime('%Y-%m-%d %H:%M:%S'))
        df.insert(len(df.columns), 'valid_to', '9999-12-31 23:59:59')
        df.insert(len(df.columns), 'valid_flag', 'Y')
        df.index.rename('id', inplace=True)

        df = df[['person_id', 'registration_number', 'membership_id', 'name', 'birthdate', 'mother_name', 'gender_id',
                 'identity_card_number', 'membership_fee_category_id', 'notes', 'valid_from', 'valid_to', 'valid_flag',
                 'organization']]
        return df


class DummyOrganizationData(DummyDataBase):
    """
    Class to create dummy data for the organization units of the NGO.
    """

    @property
    def root_organization(self) -> str:
        return self.cfg.get('organization', dict()).get('root', 'Magyar Rákellenes Liga')

    @property
    def est_min_date(self) -> datetime.date:
        return datetime.datetime.strptime(
            self.cfg.get('organization', dict()).get('min_date', "1995-01-01"),
            '%Y-%m-%d'
        ).date()

    @property
    def est_max_date(self) -> datetime.date:
        return datetime.datetime.strptime(
            self.cfg.get('organization', dict()).get('max_date', "2015-12-31"),
            '%Y-%m-%d'
        ).date()

    def generate_establishment_date(self, org_count: int) -> pd.DataFrame:
        """
        Adds randomly generated establishment dates for the organizations.
        """
        random.seed = self.seed
        return pd.DataFrame(
            data=[
                (
                    self.est_min_date + datetime.timedelta(
                        seconds=random.randint(0, int((self.est_max_date - self.est_min_date).total_seconds()))
                    )
                ).strftime('%Y-%m-%d')
                for _ in range(org_count)
            ],
            columns=['establishment_date']
        )

    def __call__(self, df_person: pd.DataFrame) -> pd.DataFrame:
        """
        Generates organization related data.

        :param df_person: DataFrame containing member related data
        """
        super().__call__()
        if 'organization' not in df_person.columns:
            raise KeyError('Column organization not found in df_person')
        df_org = df_person.loc[:, ['organization']].copy().drop_duplicates().reset_index(drop=True)
        df_org = pd.concat(
            [
                pd.DataFrame(data=[(self.root_organization,)], columns=['organization']),
                df_org.copy()
            ]
        )
        df_org.reset_index(drop=True, inplace=True)

        df_org.insert(0, 'organization_id', [x for x in range(1, len(df_org.index) + 1)])
        df_org.insert(
            1, 'organization_parent_id',
            df_org.organization.apply(lambda x: None if x == self.root_organization else 1)
        )
        df_org.insert(len(df_org.columns), 'description', None)
        df_org.insert(
            len(df_org.columns), 'accepts_members_flag',
            df_org.organization.apply(lambda x: 'N' if x == self.root_organization else 'Y')
        )
        df_org = pd.merge(df_org, self.generate_establishment_date(len(df_org.index)), left_index=True,
                          right_index=True)
        df_org.insert(len(df_org.columns), 'termination_date', None)
        df_org.insert(len(df_org.columns), 'notes', None)
        df_org.insert(len(df_org.columns), 'valid_from', self.base_date.strftime('%Y-%m-%d %H:%M:%S'))
        df_org.insert(len(df_org.columns), 'valid_to', '9999-12-31 23:59:59')
        df_org.insert(len(df_org.columns), 'valid_flag', 'Y')

        df_org.rename(columns={'organization': 'name'}, inplace=True)
        df_org.index.rename('id', inplace=True)
        return df_org


class DummyAddressData(DummyDataBase):
    """
    Class to create dummy data for addresses (including both people and organizations).
    """

    @property
    def settlement_data(self) -> pd.DataFrame:
        return self.src.get('settlement_data', pd.DataFrame()).copy()

    @property
    def street_names(self) -> pd.DataFrame:
        return self.src.get('street_names', pd.DataFrame()).copy()

    @property
    def settlement_correction(self) -> Dict[str, str]:
        return self.cfg.get('address', dict()).get('settlement_correction', dict()).copy()

    @property
    def address_type_id_map(self) -> Dict[str, int]:
        return self.cfg.get('address', dict()).get('address_type_id_map', dict()).copy()

    def get_zip_code_map(self) -> DefaultDict[str, List[str]]:
        """
        Generate a dict of settlement names with list of zip codes.
        """
        zip_map = defaultdict(list)
        for idx, row in self.settlement_data[['settlement_name', 'zip_code']].iterrows():
            zip_map[row['settlement_name']].append(str(row['zip_code']))

        return zip_map

    def map_settlements(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to map organization names to settlement names (do any necessary correction on names in order to match).

        :param df: organization or person data
        :return: extended DataFrame
        """
        df.insert(
            len(df.columns),
            'organization_correction',
            df.organization_name.apply(lambda x: self.settlement_correction.get(x, x))
        )
        df_settlements = self.settlement_data[['settlement_name']].drop_duplicates()
        df = df.merge(right=df_settlements, how='left', left_on='organization_correction', right_on='settlement_name')
        df.drop(columns=['organization_correction'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df

    def generate_zip_codes(self, df: pd.DataFrame, zip_map: DefaultDict[str, List[str]]) -> pd.DataFrame:
        """
        Method to add zip codes to settlements. In case multiple zip codes are available picks a random one.

        :param df: organization or person data, with settlement name column
        :param zip_map: mapping containing the settlement names and zip codes
        :return: mapped DataFrame
        :raises KeyError: if settlement_name column is missing
        """
        if 'settlement_name' not in df.columns:
            raise KeyError('Column settlement_name is missing from input DataFrame!')

        random.seed = self.seed
        df.insert(
            len(df.columns),
            'zip',
            df['settlement_name'].apply(lambda x: random.choice(zip_map.get(x, ['XXXX'])))
        )
        return df

    def generate_addresses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to generate fake street names and house numbers.

        :param df: concatenated organization and person address data
        :return: extended DataFrame
        """
        rows = len(df.index)

        streets: pd.DataFrame = self.street_names.sample(
            n=rows, weights='street_count', random_state=self.seed).reset_index(drop=True).iloc[:, :1]
        random.seed = self.seed
        streets.insert(len(streets.columns), 'house_number', [f'{random.randint(1, 101)}.' for _ in range(rows)])
        streets.insert(len(streets.columns), 'address_1', streets['street_name'] + ' ' + streets['house_number'])
        streets.insert(len(streets.columns), 'address_2', None)

        df = df.merge(
            right=streets,
            left_index=True,
            right_index=True,
        )
        df.drop(columns=['street_name', 'house_number'], inplace=True)
        return df

    def generate_address_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to generate address type mapping based on config provided mapping values.

        :param df: organization and person address data
        :return: extended DataFrame
        """
        df.insert(len(df.columns), 'address_type_id', self.address_type_id_map.get("Állandó lakcím", -1))
        df.loc[~df['organization_id'].isna(), 'address_type_id'] = self.address_type_id_map.get("Hivatalos cím", -1)
        df.loc[df['organization_id'] == 1.0, 'address_type_id'] = self.address_type_id_map.get("Székhely cím", -1)
        df.insert(len(df.columns), 'ref_address_type_id', None)
        return df

    def __call__(self, df_person: pd.DataFrame, df_organization: pd.DataFrame) -> pd.DataFrame:
        """
        Generates address data for both organizations and members.

        :param df_person: DataFrame with member data
        :param df_organization: DataFrame with organization data
        """
        super().__call__()
        zip_code_map = self.get_zip_code_map()

        df_org = df_organization[['organization_id', 'name', 'valid_from', 'valid_to', 'valid_flag']].copy()
        df_org.rename(columns={'name': 'organization_name'}, inplace=True)
        df_org = self.map_settlements(df_org)
        df_org = self.generate_zip_codes(df_org, zip_code_map)

        df_ppl = df_person[['person_id', 'organization', 'valid_from', 'valid_to', 'valid_flag']].copy()
        df_ppl.rename(columns={'organization': 'organization_name'}, inplace=True)
        df_ppl = self.map_settlements(df_ppl)
        df_ppl = self.generate_zip_codes(df_ppl, zip_code_map)

        df = pd.concat([df_org, df_ppl])
        df.rename(columns={'settlement_name': 'city'}, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = self.generate_addresses(df)
        df = self.generate_address_types(df)
        df = df[
            ['person_id', 'organization_id', 'address_type_id', 'ref_address_type_id', 'zip', 'city', 'address_1',
             'address_2', 'valid_from', 'valid_to', 'valid_flag']
        ]
        df.index.rename('id', inplace=True)
        return df


class DummyPhoneData(DummyDataBase):
    """
    Class to create dummy data for phone numbers (including both people and organizations).
    """

    @property
    def mobile_phone_area_codes(self) -> pd.DataFrame:
        return self.src.get('mobile_phone_area_codes').copy()

    @property
    def landline_phone_area_codes(self) -> pd.DataFrame:
        df = self.src.get('landline_phone_area_codes').copy()
        df['zip_code'] = df['zip_code'].map(str)
        return df

    def generate_mobile_phone_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to generate mobile phone numbers and corresponding messaging platform flags as well.

        :param df: organization and person id data
        :return: extended DataFrame
        """
        msg_flags = ['Y', 'N']
        rows = len(df.index)
        df_tmp: pd.DataFrame = self.mobile_phone_area_codes.sample(
            n=rows, replace=True, weights='weight', random_state=self.seed
        ).reset_index(drop=True).iloc[:, :1]
        df_tmp.index += 1
        df_tmp.insert(0, 'country_code', '+36')
        random.seed = self.seed
        df_tmp.insert(len(df_tmp.columns), 'phone',
                      [str(random.randint(0, 9999999)).zfill(7) for _ in range(rows)])
        df_tmp.insert(len(df_tmp.columns), 'phone_number',
                      df_tmp['country_code'] + '-' + df_tmp['area_code'].map(str) + '-' + df_tmp['phone'])
        df_tmp.insert(len(df_tmp.columns), 'phone_extension', None)

        for idx, row in self.messaging_platform_data.iterrows():
            df_tmp.insert(
                len(df_tmp.columns),
                row['platform_name'],
                [random.choice(msg_flags) if row['phone_flag'] == 'Y' else 'N' for _ in range(rows)]
            )

        df_tmp.drop(columns=['country_code', 'area_code', 'phone'], inplace=True)
        df_tmp = df.merge(right=df_tmp, left_index=True, right_index=True)
        df_tmp.insert(3, 'phone_type_id', 1)
        df_tmp.insert(4, 'ref_phone_type_id', None)
        return df_tmp

    def generate_landline_phone_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to generate landline phone numbers based on zip code.

        :param df: organization and person id data and zip codes
        :return: extended DataFrame
        """
        df_land = df.merge(right=self.landline_phone_area_codes, how='left', left_on='zip', right_on='zip_code')
        df_land.loc[df_land['area_code'].isnull(), 'area_code'] = 1
        df_land['area_code'] = df_land['area_code'].map(int).map(str)
        df_land.insert(len(df_land.columns), 'country_code', '+36')
        random.seed = self.seed
        df_land.insert(
            len(df_land.columns),
            'phone',
            df_land['area_code'].apply(
                lambda x: str(random.randint(0, 9999999)).zfill(7) if x == '1'
                else str(random.randint(0, 999999)).zfill(6)
            )
        )
        df_land.insert(
            len(df_land.columns),
            'phone_number',
            df_land['country_code'] + '-' + df_land['area_code'] + '-' + df_land['phone']
        )
        df_land.insert(len(df_land.columns), 'phone_extension', None)

        for idx, row in self.messaging_platform_data.iterrows():
            df_land.insert(len(df_land.columns), row['platform_name'], 'N')

        df_land.insert(3, 'phone_type_id', 2)
        df_land.insert(4, 'ref_phone_type_id', None)

        return df_land.drop(
            columns=['settlement', 'county', 'micro_region', 'area_code', 'zip_code', 'country_code', 'phone']
        )

    def __call__(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generates phone number data.

        :param df_input: DataFrame containing organization and people data (id fields and zip code). In case of
                         multiple zip codes first is kept only!
        :raise KeyError: in case person_id, organization_id or zip column is missing
        """
        super().__call__()
        required_columns = ['person_id', 'organization_id', 'zip']
        for col in required_columns:
            if col not in df_input.columns:
                raise KeyError(f'Missing column: {col}')

        df = df_input[required_columns].drop_duplicates(subset=required_columns)
        df_mobile = self.generate_mobile_phone_numbers(df)
        df_landline = self.generate_landline_phone_numbers(df)

        df_phone: pd.DataFrame = pd.concat([df_mobile, df_landline]).reset_index(drop=True)
        df_phone.drop(columns=['zip'], inplace=True)
        df_phone.insert(len(df_phone.columns), 'valid_from', self.base_date.strftime('%Y-%m-%d %H:%M:%S'))
        df_phone.insert(len(df_phone.columns), 'valid_to', '9999-12-31 23:59:59')
        df_phone.insert(len(df_phone.columns), 'valid_flag', 'Y')
        df_phone.index.rename('id', inplace=True)
        return df_phone


class DummyEmailData(DummyDataBase):
    """
    Class to create dummy data for e-mail addresses (including both people and organizations).
    """

    @property
    def email_service_providers(self) -> pd.DataFrame:
        return self.src.get('email_service_providers').copy()

    @property
    def non_ascii(self) -> Dict[str, str]:
        return {
            'á': 'a',
            'é': 'e',
            'í': 'i',
            'ó': 'o',
            'ö': 'o',
            'ő': 'o',
            'õ': 'o',
            'ú': 'u',
            'ü': 'u',
            'ű': 'u',
        }

    def convert_to_ascii(self, word: str) -> str:
        """
        Helper function to convert Hungarian letters to ASCII characters.

        :param word: word which might contain Hungarian letters
        :return: cleaned up word
        """
        return ''.join([self.non_ascii.get(char, char) for char in word])

    def generate_personal_email(self, df_person: pd.DataFrame) -> pd.DataFrame:
        """
        Method to generate e-mail addresses for members.

        :param df_person: DataFrame with member data
        :return: DataFrame with person_id and e-mail related data
        """
        msg_flags = ['Y', 'N']
        rows = len(df_person.index)
        email_providers = self.email_service_providers.sample(
            n=rows, replace=True, weights='weight', random_state=self.seed
        ).reset_index(drop=True).iloc[:, :1]
        email_providers.index += 1

        df = df_person[['person_id', 'name']].copy()
        df.insert(len(df.columns), 'email_type_id', 1)
        df.insert(len(df.columns), 'ref_email_type_id', None)
        df.insert(len(df.columns), 'email',
                  df['name'].apply(lambda x: x.lower().replace(' ', '.')) + '@' + email_providers['provider'])
        df['email'] = df['email'].apply(self.convert_to_ascii)

        msg_platform = self.messaging_platform_data.loc[self.messaging_platform_data['email_flag'] == 'Y']
        for idx, row in msg_platform.iterrows():
            df.insert(
                len(df.columns),
                row['platform_name'],
                [random.choice(msg_flags) for _ in range(rows)]
            )
        return df

    def generate_organization_email(self, df_org: pd.DataFrame) -> pd.DataFrame:
        """
        Method to generate e-mail addresses for organizations.

        :param df_org: DataFrame with organization data
        :return: DataFrame with organization_id and e-mail related data
        """
        df = df_org[['organization_id', 'name']].copy()
        df.insert(len(df.columns), 'email_type_id', 1)
        df.insert(len(df.columns), 'ref_email_type_id', None)
        df.insert(len(df.columns), 'email', df['name'].apply(lambda x: x.lower().replace(' ', '.')) + '@rakliga.hu')
        df['email'] = df['email'].apply(self.convert_to_ascii)

        msg_platform = self.messaging_platform_data.loc[self.messaging_platform_data['email_flag'] == 'Y']
        for idx, row in msg_platform.iterrows():
            df.insert(len(df.columns), row['platform_name'], 'N')
        return df

    def __call__(self, df_person: pd.DataFrame, df_org: pd.DataFrame) -> pd.DataFrame:
        """
        Generates e-mail address data (both for organizations and people).

        :param df_person: DataFrame with member data
        :param df_org: DataFrame with organization data
        :raise KeyError: in case person_id, organization_id or zip column is missing
        """
        super().__call__()
        required_cols_ppl = ['person_id', 'name']
        required_cols_org = ['organization_id', 'name']
        for col in required_cols_ppl:
            if col not in df_person.columns:
                raise KeyError(f'Missing column: {col} from df_person')
        for col in required_cols_org:
            if col not in df_org.columns:
                raise KeyError(f'Missing column: {col} from df_org')
        df_email_ppl = self.generate_personal_email(df_person)
        df_email_org = self.generate_organization_email(df_org)

        df_email: pd.DataFrame = pd.concat([df_email_org, df_email_ppl]).reset_index(drop=True)
        df_email = df_email[['person_id', 'organization_id', 'email_type_id', 'ref_email_type_id', 'email',
                             'messenger', 'skype']]

        df_email.insert(len(df_email.columns), 'valid_from', self.base_date.strftime('%Y-%m-%d %H:%M:%S'))
        df_email.insert(len(df_email.columns), 'valid_to', '9999-12-31 23:59:59')
        df_email.insert(len(df_email.columns), 'valid_flag', 'Y')
        df_email.index.rename('id', inplace=True)
        return df_email


class DummyMembershipData(DummyDataBase):
    """
    Class to create dummy data for relation between people and organizations.
    """

    def __call__(self, df_person: pd.DataFrame, df_org: pd.DataFrame) -> pd.DataFrame:
        """
        Generates membership data based on initial mapping of people and organizations.

        :param df_person:
        :param df_org:
        """
        df_membership = pd.merge(left=df_person, right=df_org, how='left', left_on='organization', right_on='name')
        df_membership = df_membership[['person_id', 'organization_id']].copy()
        df_membership.insert(len(df_membership.columns), 'active_flag', 'Y')
        df_membership.insert(len(df_membership.columns), 'inactivity_status_id', None)
        df_membership.insert(len(df_membership.columns), 'event_date', self.base_date.strftime('%Y-%m-%d %H:%M:%S'))
        df_membership.insert(len(df_membership.columns), 'notes', None)
        df_membership.insert(len(df_membership.columns), 'valid_from', self.base_date.strftime('%Y-%m-%d %H:%M:%S'))
        df_membership.insert(len(df_membership.columns), 'valid_to', '9999-12-31 23:59:59')
        df_membership.insert(len(df_membership.columns), 'valid_flag', 'Y')
        df_membership.index.rename('id', inplace=True)
        return df_membership
