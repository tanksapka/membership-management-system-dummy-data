import getpass
import logging
import os
from typing import Optional
from urllib.parse import quote_plus

_logger = logging.getLogger(__name__)


def get_odbc_connection_string() -> str:
    """
    Ask user input for database credentials.
    In case of server, database and driver it tries to look up environment variables first.
    server: MS_SQL_SERVER
    database: MS_SQL_DB
    driver: SQL_NATIVE_CLIENT

    :return: pyodbc connection string
    :raises ValueError: in case of empty user input
    """
    if "MS_SQL_SERVER" in os.environ:
        server = os.environ["MS_SQL_SERVER"]
    else:
        server_input = input("Server address: ")
        if not server_input:
            raise ValueError("Please provide a valid server name!")
        server = server_input

    if "MS_SQL_DB" in os.environ:
        database = os.environ["MS_SQL_DB"]
    else:
        db_input = input("Database name: ")
        if not db_input:
            raise ValueError("Please provide a valid database name!")
        database = db_input

    if "SQL_NATIVE_CLIENT" in os.environ:
        driver = os.environ["SQL_NATIVE_CLIENT"]
    else:
        driver_input = input("Driver name: ")
        if not driver_input:
            raise ValueError("Please provide a valid database driver name!")
        driver = driver_input

    uid_input = input("User_id: ")
    if not uid_input:
        raise ValueError("Please provide a valid user id!")
    uid = uid_input

    pwd_input = getpass.getpass()
    if not pwd_input:
        raise ValueError("Please provide a valid password!")
    pwd = pwd_input

    return f"DRIVER={{{driver}}};SERVER=tcp:{server},1433;DATABASE={database};UID={uid};PWD={pwd};Encrypt=yes;" \
           f"TrustServerCertificate=no;Connection Timeout=30;"


def get_sqlalchemy_connection_string(odbc_conn_str: Optional[str] = None) -> str:
    """
    Generates sqlalchemy compatible connection string. If odbc_conn_str is not provided, calls the function which
    generates it.

    :param odbc_conn_str: pyodbc connection string
    :return: sqlalchemy url string
    """
    return f'mssql+pyodbc:///?odbc_connect={quote_plus(odbc_conn_str or get_odbc_connection_string())}'
