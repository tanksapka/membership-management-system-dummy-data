import os
import unittest
from data_generator.web_data import BaseScraper, FamilyNameScraper
from unittest.mock import patch, MagicMock


class TestBaseScraper(unittest.TestCase):

    @patch('data_generator.web_data.os')
    def test_get_proxies(self, mock_os):
        mock_os.environ = {
            'HTTP_PROXY': 'foo',
            'HTTPS_PROXY': 'bar',
        }
        bs = BaseScraper()
        self.assertDictEqual(bs.get_proxies(), {'http': 'foo', 'https': 'bar'})

        mock_os.environ = {
            'HTTP_PROXY': 'foo',
        }
        self.assertDictEqual(bs.get_proxies(), {'http': 'foo'})


class TestFamilyNameScraper(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'family_name_test.txt')):
            os.remove(os.path.join(os.path.dirname(__file__), 'family_name_test.txt'))

    @patch('data_generator.web_data.os')
    @patch('data_generator.web_data.requests')
    def test_get_initials(self, mock_requests, mock_os):
        mock_os.environ = {
            'HTTP_PROXY': 'foo',
            'HTTPS_PROXY': 'bar',
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        with open(os.path.join(os.path.dirname(__file__), 'fixtures', 'initials.html'), 'r', encoding='utf-8') as fh:
            mock_response.text = fh.read()
        mock_requests.get.return_value = mock_response

        with patch('data_generator.web_data._logger', autospec=True):
            fns = FamilyNameScraper()
            results = fns.get_initials()
            self.assertListEqual(results, ['http://vagyok.net/vezeteknevek/a/'])
            mock_requests.get.assert_called_once()

            mock_response.status_code = 404
            self.assertListEqual(fns.get_initials(), list())

    @patch('data_generator.web_data.os')
    @patch('data_generator.web_data.requests')
    def test_get_family_names(self, mock_requests, mock_os):
        mock_os.environ = {
            'HTTP_PROXY': 'foo',
            'HTTPS_PROXY': 'bar',
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        result_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'family_names_a.html')
        with open(result_path, 'r', encoding='utf-8') as fh:
            mock_response.text = fh.read()
        mock_requests.get.return_value = mock_response

        with patch('data_generator.web_data._logger', autospec=True):
            fns = FamilyNameScraper()
            results = fns.get_family_names("http://vagyok.net/vezeteknevek/a/")
            self.assertEqual(len(results), 636)
            self.assertTrue('Asztalos' in results)
            mock_requests.get.assert_called_once()

            mock_response.status_code = 404
            self.assertListEqual(fns.get_family_names("http://vagyok.net/vezeteknevek/a/"), list())

    @patch('data_generator.web_data.os')
    @patch('data_generator.web_data.requests')
    def test_execute(self, mock_requests, mock_os):
        mock_os.environ = {
            'HTTP_PROXY': 'foo',
            'HTTPS_PROXY': 'bar',
        }
        result_paths = [
            os.path.join(os.path.dirname(__file__), 'fixtures', 'initials.html'),
            os.path.join(os.path.dirname(__file__), 'fixtures', 'family_names_a.html'),
        ]
        responses = list()
        for path in result_paths:
            mock_response = MagicMock()
            mock_response.status_code = 200
            with open(path, 'r', encoding='utf-8') as fh:
                mock_response.text = fh.read()
            responses.append(mock_response)
        mock_requests.get.side_effect = responses

        with patch('data_generator.web_data._logger', autospec=True):
            fns = FamilyNameScraper()
            results = fns.execute(os.path.join(os.path.dirname(__file__), 'family_name_test.txt'))
            self.assertEqual(len(results), 636)
            self.assertTrue('Asztalos' in results)
            mock_requests.get.assert_called()
            self.assertEqual(mock_requests.get.call_count, 2)


if __name__ == '__main__':
    unittest.main()
