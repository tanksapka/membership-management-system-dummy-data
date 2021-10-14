"""
This module creates the dummy data replicating the memberdb data model and exports it into csv files.
"""
import datetime
import os
import logging
import pandas as pd
import random
import toml
import data_generator.source_data as sd
from string import ascii_uppercase, digits
from inspect import getmembers, isfunction
from logging import config
from typing import Dict, List, Any


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
    def __init__(self, output_folder: str) -> None:
        _logger.info('Initializing DummyData class')
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            _logger.info('Creating output folder (%s) as it does not exist' % self.output_folder)
            os.makedirs(self.output_folder)

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

    def _load_config(self) -> None:
        py_project = toml.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pyproject.toml'))
        self._config = py_project.get('dummy', dict()).get('config', dict())

    @property
    def config(self) -> Dict[str, Any]:
        return self._config.copy()

    def generate_dummy_data(self) -> List[str]:
        """
        Main method to generate the dummy data for the database's data model. Returns the list of output file paths.
        """
        pass


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
        random.seed(self.seed)
        days = [
            int(trunc_gauss(**{k: v * 365 for k, v in self.age_distribution.items()}))
            for _ in range(count)
        ]
        return pd.DataFrame(
            [(self.base_date - datetime.timedelta(days=day)).strftime('%Y-%m-%d') for day in days],
            columns=['date_of_birth']
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

    def generate_identity_card_numbers(self, count: int) -> pd.DataFrame:
        """
        Generates a dataframe of id card numbers
        """
        random.seed(self.seed)
        return pd.DataFrame(
            data=[
                ''.join(random.choices(digits, k=6)) + ''.join(random.choices(ascii_uppercase, k=2))
                for _ in range(count)
            ],
            columns=['identity_card_number'],
        )

    def map_membership_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds membership category codes based on age. Manipulates directly the provided DataFrame!
        """
        df.insert(len(df.columns), 'membership_category', self.membership_id_map.get(-1))
        for age, id_ in self.membership_id_map.items():
            mask = (df.date_of_birth
                      .apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
                      .apply(lambda x: (self.base_date - x.date()).days) <= age * 365)
            df.loc[mask, 'membership_category'] = id_
        return df

    def __call__(self) -> pd.DataFrame:
        """
        Runs data generation for person data. Output dataframe will also contain the organization where the given member
        belongs to.
        """
        df = self.generate_first_names()
        df = pd.merge(df, self.generate_last_names(len(df)), left_index=True, right_index=True)
        df.insert(len(df.columns), 'name', df['last_name'] + ' ' + df['first_name'])

        df = pd.merge(df, self.generate_date_of_birth(len(df)), left_index=True, right_index=True)
        df = pd.merge(df, self.generate_mother_name(len(df)), left_index=True, right_index=True)
        df = pd.merge(df, self.generate_identity_card_numbers(len(df)), left_index=True, right_index=True)
        self.map_membership_category(df)

        df.insert(0, 'person_id', [x for x in range(1, len(df) + 1)])
        df.insert(1, 'registration_number', df['person_id'])
        df.insert(2, 'membership_id', df['person_id'].apply(lambda x: 'A{:9>6}'.format(x)))
        df.insert(len(df.columns), 'notes', None)
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
        random.seed(self.seed)
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
        if 'organization' not in df_person.columns:
            raise KeyError('Column organization not found in df_person')
        df_org = df_person.loc[:, ['organization']].copy().drop_duplicates().reset_index(drop=True)
        df_org = pd.concat(
            [
                pd.DataFrame(data=[(self.root_organization,)], columns=['organization']),
                df_org.copy()
            ]
        )

        df_org.insert(0, 'id', [x for x in range(1, len(df_org) + 1)])
        df_org.insert(1, 'organization_id', [x for x in range(1, len(df_org) + 1)])
        df_org.insert(
            2, 'organization_parent_id',
            df_org.organization.apply(lambda x: None if x == self.root_organization else 1)
        )
        df_org.insert(len(df_org.columns), 'description', None)
        df_org.insert(
            len(df_org.columns), 'accepts_members_flag',
            df_org.organization.apply(lambda x: 'N' if x == self.root_organization else 'Y')
        )
        df_org = pd.merge(df_org, self.generate_establishment_date(len(df_org)), left_index=True, right_index=True)
        df_org.insert(len(df_org.columns), 'termination_date', None)
        df_org.insert(len(df_org.columns), 'notes', None)
        df_org.insert(len(df_org.columns), 'valid_from', self.base_date.strftime('%Y-%m-%d %H:%M:S'))
        df_org.insert(len(df_org.columns), 'valid_to', '9999-12-31 23:59:59')
        df_org.insert(len(df_org.columns), 'valid_flag', 'Y')

        df_org.rename(columns={'organization': 'name'}, inplace=True)
        return df_org


class DummyAddressData(DummyDataBase):
    """
    Class to create dummy data for addresses (including both people and organizations).
    """

    @property
    def settlement_data(self) -> pd.DataFrame:
        return self.src.get('settlement_data', pd.DataFrame()).copy()

    @property
    def settlement_correction(self) -> Dict[str, str]:
        return self.cfg.get('address', dict()).get('settlement_correction', dict()).copy()

    def map_settlements(self, df: pd.DataFrame) -> pd.DataFrame:
        df.insert(
            len(df.columns),
            'organization_correction',
            df.organization.apply(lambda x: self.settlement_correction.get(x, 'N/A'))
        )
        df_settlements = self.settlement_data[
            ['settlement_name', 'district_code', 'district_name', 'resident_population']
        ].drop_duplicates()
        df.merge(right=df_settlements, how='left', left_on='organization_correction', right_on='settlement_name')
        df.drop(columns=['organization_correction'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df

    def __call__(self, df_person: pd.DataFrame, df_org: pd.DataFrame) -> pd.DataFrame:
        # Map organization name to settlement name, non-matching should go to Budapest
        # Get the set of districts based on the settlement coverage and corresponding member count
        # Based on resident population distribute members along the settlements - do this based on ZIP code
        # For organizations use the first element (if there is multiple) from ZIP codes. # TODO: or random?

        df_ppl = df_person[['organization', 'membership_id']].groupby(by=['organization'], as_index=False).count()
        df_ppl = self.map_settlements(df_ppl)


        pass


class DummyPhoneData(DummyDataBase):
    """
    Class to create dummy data for phone numbers (including both people and organizations).
    """
    def __call__(self, *args, **kwargs):
        pass


class DummyEmailData(DummyDataBase):
    """
    Class to create dummy data for e-mail addresses (including both people and organizations).
    """
    def __call__(self, *args, **kwargs):
        pass
