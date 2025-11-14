"""Data Connectors Module"""

from llm_copilot.connectors.database import (
    DatabaseConnector,
    PostgreSQLConnector,
    MySQLConnector,
    SnowflakeConnector,
    DatabricksConnector,
    BigQueryConnector,
    SQLiteConnector,
    SQLServerConnector,
    OracleConnector,
    RedshiftConnector,
    DuckDBConnector,
    ParquetConnector
)
from llm_copilot.connectors.file import (
    FileConnector,
    CSVConnector,
    ExcelConnector,
    JSONConnector,
    GoogleSheetsConnector,
    ParquetFileConnector
)
from llm_copilot.connectors.deepcausalmmm import DeepCausalMMMConnector

__all__ = [
    # Main connectors (auto-detect)
    'DatabaseConnector',
    'FileConnector',
    'DeepCausalMMMConnector',
    
    # Specific database connectors
    'PostgreSQLConnector',
    'MySQLConnector',
    'SnowflakeConnector',
    'DatabricksConnector',
    'BigQueryConnector',
    'SQLiteConnector',
    'SQLServerConnector',
    'OracleConnector',
    'RedshiftConnector',
    'DuckDBConnector',
    'ParquetConnector',
    
    # File connectors
    'CSVConnector',
    'ExcelConnector',
    'JSONConnector',
    'GoogleSheetsConnector',
    'ParquetFileConnector'
]

