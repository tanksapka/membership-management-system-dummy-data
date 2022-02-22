"""
Support module to fetch the missing source data from the web.
"""
import os
import logging
import re
import requests
import time
import toml
from bs4 import BeautifulSoup
from logging import config
from typing import List, Dict
from urllib.parse import urljoin


log_cfg = toml.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pyproject.toml'))
config.dictConfig(log_cfg)
_logger = logging.getLogger(__name__)


class BaseScraper:
    """
    Base class to house common methods and properties for scraping.
    """
    def __init__(self) -> None:
        self.proxy_dict = self.get_proxies()

    @staticmethod
    def get_proxies() -> Dict[str, str]:
        proxies = {
            'http': os.environ.get('HTTP_PROXY'),
            'https': os.environ.get('HTTPS_PROXY'),
        }
        return {k: v for k, v in proxies.items() if v}


class FamilyNameScraper(BaseScraper):
    """
    Collect all the available Hungarian family names from http://vagyok.net url.
    """
    def __init__(self) -> None:
        super().__init__()
        self.site_url = 'http://vagyok.net/'

    def get_initials(self) -> List[str]:
        """
        Method to fetch urls for family name pages (e.g. http://vagyok.net/vezeteknevek/a/).
        """
        _logger.info('Fetching family name initials')
        url = urljoin(self.site_url, 'vezeteknevek/')
        response = requests.get(url, proxies=self.proxy_dict)
        if response.status_code != 200:
            _logger.warning('Fetching failed with response status code: %d' % response.status_code)
            return list()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all(
            lambda tag: tag.has_attr('href') and 'vezeteknevek' in tag['href'] and tag.has_attr('rel')
        )
        _logger.info('Fetching done, %d initials found' % len(links))
        return [urljoin(self.site_url, link['href']) for link in links]

    def get_family_names(self, url: str) -> List[str]:
        """
        Collects the list of family names from the given url. The url should be a landing page for a specific initial
        (e.g. http://vagyok.net/vezeteknevek/a/).
        """
        _logger.info('Fetching family names from: %s' % url)
        response = requests.get(url, proxies=self.proxy_dict)
        if response.status_code != 200:
            _logger.warning('Fetching failed with response status code: %d' % response.status_code)
            return list()
        soup = BeautifulSoup(response.text, 'html.parser')
        re_href_value = re.compile(r'/vezeteknevek/[\w-]{2,}/', re.IGNORECASE)
        links = soup.find_all(lambda tag: tag.has_attr('href') and re.match(re_href_value, tag['href']))
        _logger.info('Fetching done, %d family names found on %s' % (len(links), url))
        return [link.text.capitalize() for link in links]

    def execute(self, output_path: str) -> List[str]:
        """
        Run the scraper and collect the family names. Output path should include the filename as well, already existing
        file is going to be overwritten.
        """
        _logger.info('Running scraper on family names')
        initials_urls = self.get_initials()
        results = list()
        for initial_url in initials_urls:
            results.extend(self.get_family_names(initial_url))
            time.sleep(5)

        _logger.info('Scraping finished, retrieved %d family names' % len(results))
        _logger.info('Writing results to output file: %s' % output_path)
        with open(output_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(results))

        return results
