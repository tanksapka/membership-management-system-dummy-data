import os
import pandas as pd
import unittest
from data_generator.source_data import get_src_path, load_email_service_providers, load_mobile_phone_area_codes, \
    load_landline_phone_area_codes, load_messaging_platform_data, load_settlement_data, load_first_names, \
    _load_most_common_last_names, _load_enriched_last_names, load_last_names, load_organization_stats
from unittest.mock import patch


class TestSourceData(unittest.TestCase):

    def test_get_src_path(self):
        tst_file = 'osszesffi.txt'
        tst_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', tst_file)

        self.assertEqual(get_src_path(tst_file), tst_path)
        self.assertRaises(FileNotFoundError, get_src_path, 'foo_bar.txt')

    def test_load_email_service_providers(self):
        with patch('data_generator.source_data._logger', autospec=True):
            result = load_email_service_providers()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), ['provider', 'weight'])

    def test_load_mobile_phone_area_codes(self):
        with patch('data_generator.source_data._logger', autospec=True):
            result = load_mobile_phone_area_codes()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), ['area_code', 'service_provider', 'weight'])

    def test_load_landline_phone_area_codes(self):
        with patch('data_generator.source_data._logger', autospec=True):
            result = load_landline_phone_area_codes()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), ['settlement', 'county', 'micro_region', 'area_code', 'zip_code'])

    def test_load_messaging_platform_data(self):
        with patch('data_generator.source_data._logger', autospec=True):
            result = load_messaging_platform_data()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), ['platform_name', 'phone_flag', 'email_flag'])

    def test_load_settlement_data(self):
        with patch('data_generator.source_data._logger', autospec=True):
            result = load_settlement_data()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), [
            'settlement_name', 'zip_code', 'part_of_settlement', 'settlement_ksh_code', 'settlement_legal_status',
            'county_name', 'district_code', 'district_name', 'resident_population',
        ])

    @patch('data_generator.source_data.os')
    def test_load_first_names(self, mock_os):
        mock_os.path.join.side_effect = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'osszesffi.txt'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'osszesnoi.txt'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'foo_bar.txt'),
        ]
        mock_os.path.exists.side_effect = [True, True, False]

        with patch('data_generator.source_data._logger', autospec=True):
            result = load_first_names()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), ['first_name', 'gender'])

        with patch('data_generator.source_data._logger', autospec=True):
            self.assertRaises(FileNotFoundError, load_first_names)

    def test__load_most_common_last_names(self):
        with patch('data_generator.source_data._logger', autospec=True):
            result = _load_most_common_last_names()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), ['last_name'])

    def test__load_enriched_last_names_simple(self):
        with patch('data_generator.source_data._logger', autospec=True):
            result = _load_enriched_last_names()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), ['last_name'])

    @patch('data_generator.source_data.os')
    @patch('data_generator.source_data.FamilyNameScraper')
    def test__load_enriched_last_names_scrape(self, mock_scraper, mock_os):
        mock_os.path.exists.return_value = False
        mock_scraper.execute.return_value = ['Asztalos']

        with patch('data_generator.source_data._logger', autospec=True):
            result = _load_enriched_last_names()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), ['last_name'])

    def test_load_last_names(self):
        with patch('data_generator.source_data._logger', autospec=True):
            result = load_last_names()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), ['last_name', 'most_common_flag'])

    def test_load_organization_stats(self):
        with patch('data_generator.source_data._logger', autospec=True):
            result = load_organization_stats()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), [
            'organization', 'members_2018', 'members_2019', 'members_2020', 'members_2021'
        ])


if __name__ == '__main__':
    unittest.main()
