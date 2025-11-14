"""
Database Connectors

Load MMM data from various database sources.
"""

from typing import Optional, Dict, List
from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Base class for database connectors.
    
    Supports:
    - PostgreSQL
    - MySQL
    - Snowflake
    - Databricks
    - BigQuery
    
    Examples
    --------
    >>> connector = DatabaseConnector.from_url("postgresql://user:pass@host:5432/db")
    >>> data = connector.load_mmm_data(schema="mmm_schema", table="mmm_results")
    """
    
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.connection = None
        logger.info(f"Initialized DatabaseConnector")
    
    @classmethod
    def from_url(cls, connection_url: str) -> 'DatabaseConnector':
        """Create connector from connection URL"""
        if 'postgresql' in connection_url or 'postgres' in connection_url:
            return PostgreSQLConnector(connection_url)
        elif 'mysql' in connection_url:
            return MySQLConnector(connection_url)
        elif 'snowflake' in connection_url:
            return SnowflakeConnector(connection_url)
        elif 'databricks' in connection_url:
            return DatabricksConnector(connection_url)
        elif 'bigquery' in connection_url:
            return BigQueryConnector(connection_url)
        elif 'sqlite' in connection_url:
            return SQLiteConnector(connection_url)
        elif 'mssql' in connection_url or 'sqlserver' in connection_url:
            return SQLServerConnector(connection_url)
        elif 'oracle' in connection_url:
            return OracleConnector(connection_url)
        elif 'redshift' in connection_url:
            return RedshiftConnector(connection_url)
        elif 'duckdb' in connection_url:
            return DuckDBConnector(connection_url)
        else:
            raise ValueError(f"Unsupported database: {connection_url}")
    
    def connect(self) -> None:
        """Establish database connection"""
        raise NotImplementedError
    
    def load_mmm_data(
        self,
        *,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Load MMM data from database.
        
        Parameters
        ----------
        schema : str, optional
            Database schema
        table : str, optional
            Table name
        query : str, optional
            Custom SQL query (overrides schema/table)
        filters : Dict, optional
            Filters to apply (e.g., {'channel': ['TV', 'Radio']})
            
        Returns
        -------
        pd.DataFrame
            MMM data
        """
        raise NotImplementedError
    
    def close(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Closed database connection")


class PostgreSQLConnector(DatabaseConnector):
    """PostgreSQL connector"""
    
    def connect(self) -> None:
        try:
            import psycopg2
        except ImportError:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
        
        self.connection = psycopg2.connect(self.connection_url)
        logger.info("Connected to PostgreSQL")
    
    def load_mmm_data(
        self,
        *,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        if not self.connection:
            self.connect()
        
        # Build query
        if query:
            sql = query
        else:
            if not schema or not table:
                raise ValueError("Must provide either query or (schema + table)")
            sql = f"SELECT * FROM {schema}.{table}"
            
            if filters:
                conditions = []
                for col, values in filters.items():
                    if isinstance(values, list):
                        values_str = ', '.join(f"'{v}'" for v in values)
                        conditions.append(f"{col} IN ({values_str})")
                    else:
                        conditions.append(f"{col} = '{values}'")
                sql += " WHERE " + " AND ".join(conditions)
        
        logger.info(f"Executing query: {sql[:100]}...")
        df = pd.read_sql(sql, self.connection)
        logger.info(f"Loaded {len(df)} rows from PostgreSQL")
        
        return df


class MySQLConnector(DatabaseConnector):
    """MySQL connector"""
    
    def connect(self) -> None:
        try:
            import pymysql
        except ImportError:
            raise ImportError("pymysql not installed. Install with: pip install pymysql")
        
        # Parse connection URL
        import urllib.parse
        parsed = urllib.parse.urlparse(self.connection_url)
        
        self.connection = pymysql.connect(
            host=parsed.hostname,
            port=parsed.port or 3306,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.lstrip('/')
        )
        logger.info("Connected to MySQL")
    
    def load_mmm_data(
        self,
        *,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        if not self.connection:
            self.connect()
        
        # Build query (similar to PostgreSQL)
        if query:
            sql = query
        else:
            if not table:
                raise ValueError("Must provide either query or table")
            sql = f"SELECT * FROM {table}"
            
            if filters:
                conditions = []
                for col, values in filters.items():
                    if isinstance(values, list):
                        values_str = ', '.join(f"'{v}'" for v in values)
                        conditions.append(f"{col} IN ({values_str})")
                    else:
                        conditions.append(f"{col} = '{values}'")
                sql += " WHERE " + " AND ".join(conditions)
        
        df = pd.read_sql(sql, self.connection)
        logger.info(f"Loaded {len(df)} rows from MySQL")
        
        return df


class SnowflakeConnector(DatabaseConnector):
    """Snowflake connector"""
    
    def connect(self) -> None:
        try:
            import snowflake.connector
        except ImportError:
            raise ImportError("snowflake-connector-python not installed. Install with: pip install snowflake-connector-python")
        
        # Parse Snowflake connection string
        # Format: snowflake://user:password@account/database/schema?warehouse=wh
        import urllib.parse
        parsed = urllib.parse.urlparse(self.connection_url)
        query_params = dict(urllib.parse.parse_qsl(parsed.query))
        
        self.connection = snowflake.connector.connect(
            user=parsed.username,
            password=parsed.password,
            account=parsed.hostname,
            warehouse=query_params.get('warehouse'),
            database=parsed.path.split('/')[1] if len(parsed.path.split('/')) > 1 else None,
            schema=parsed.path.split('/')[2] if len(parsed.path.split('/')) > 2 else None
        )
        logger.info("Connected to Snowflake")
    
    def load_mmm_data(
        self,
        *,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        # Build query
        if query:
            sql = query
        else:
            if not schema or not table:
                raise ValueError("Must provide either query or (schema + table)")
            sql = f"SELECT * FROM {schema}.{table}"
            
            if filters:
                conditions = []
                for col, values in filters.items():
                    if isinstance(values, list):
                        values_str = ', '.join(f"'{v}'" for v in values)
                        conditions.append(f"{col} IN ({values_str})")
                    else:
                        conditions.append(f"{col} = '{values}'")
                sql += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(sql)
        df = cursor.fetch_pandas_all()
        cursor.close()
        
        logger.info(f"Loaded {len(df)} rows from Snowflake")
        return df


class DatabricksConnector(DatabaseConnector):
    """Databricks connector"""
    
    def connect(self) -> None:
        try:
            from databricks import sql
        except ImportError:
            raise ImportError("databricks-sql-connector not installed. Install with: pip install databricks-sql-connector")
        
        # Parse Databricks connection
        import urllib.parse
        parsed = urllib.parse.urlparse(self.connection_url)
        query_params = dict(urllib.parse.parse_qsl(parsed.query))
        
        self.connection = sql.connect(
            server_hostname=parsed.hostname,
            http_path=query_params.get('http_path'),
            access_token=query_params.get('token')
        )
        logger.info("Connected to Databricks")
    
    def load_mmm_data(
        self,
        *,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        # Build query
        if query:
            sql = query
        else:
            if not schema or not table:
                raise ValueError("Must provide either query or (schema + table)")
            sql = f"SELECT * FROM {schema}.{table}"
            
            if filters:
                conditions = []
                for col, values in filters.items():
                    if isinstance(values, list):
                        values_str = ', '.join(f"'{v}'" for v in values)
                        conditions.append(f"{col} IN ({values_str})")
                    else:
                        conditions.append(f"{col} = '{values}'")
                sql += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(sql)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(result, columns=columns)
        cursor.close()
        
        logger.info(f"Loaded {len(df)} rows from Databricks")
        return df


class BigQueryConnector(DatabaseConnector):
    """Google BigQuery connector"""
    
    def connect(self) -> None:
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ImportError("google-cloud-bigquery not installed. Install with: pip install google-cloud-bigquery")
        
        self.connection = bigquery.Client()
        logger.info("Connected to BigQuery")
    
    def load_mmm_data(
        self,
        *,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        if not self.connection:
            self.connect()
        
        # Build query
        if query:
            sql = query
        else:
            if not schema or not table:
                raise ValueError("Must provide either query or (schema + table)")
            sql = f"SELECT * FROM `{schema}.{table}`"
            
            if filters:
                conditions = []
                for col, values in filters.items():
                    if isinstance(values, list):
                        values_str = ', '.join(f"'{v}'" for v in values)
                        conditions.append(f"{col} IN ({values_str})")
                    else:
                        conditions.append(f"{col} = '{values}'")
                sql += " WHERE " + " AND ".join(conditions)
        
        df = self.connection.query(sql).to_dataframe()
        logger.info(f"Loaded {len(df)} rows from BigQuery")
        
        return df


class SQLiteConnector(DatabaseConnector):
    """SQLite connector (local database)"""
    
    def connect(self) -> None:
        import sqlite3
        
        # Extract path from URL (sqlite:///path/to/db.sqlite)
        db_path = self.connection_url.replace('sqlite:///', '')
        self.connection = sqlite3.connect(db_path)
        logger.info(f"Connected to SQLite: {db_path}")
    
    def load_mmm_data(
        self,
        *,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        if not self.connection:
            self.connect()
        
        # Build query
        if query:
            sql = query
        else:
            if not table:
                raise ValueError("Must provide either query or table")
            sql = f"SELECT * FROM {table}"
            
            if filters:
                conditions = []
                for col, values in filters.items():
                    if isinstance(values, list):
                        values_str = ', '.join(f"'{v}'" for v in values)
                        conditions.append(f"{col} IN ({values_str})")
                    else:
                        conditions.append(f"{col} = '{values}'")
                sql += " WHERE " + " AND ".join(conditions)
        
        df = pd.read_sql(sql, self.connection)
        logger.info(f"Loaded {len(df)} rows from SQLite")
        
        return df


class SQLServerConnector(DatabaseConnector):
    """Microsoft SQL Server / Azure SQL connector"""
    
    def connect(self) -> None:
        try:
            import pyodbc
        except ImportError:
            raise ImportError("pyodbc not installed. Install with: pip install pyodbc")
        
        # Parse connection URL
        # Format: mssql://user:pass@host:port/database
        import urllib.parse
        parsed = urllib.parse.urlparse(self.connection_url)
        
        connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={parsed.hostname},{parsed.port or 1433};"
            f"DATABASE={parsed.path.lstrip('/')};"
            f"UID={parsed.username};"
            f"PWD={parsed.password}"
        )
        
        self.connection = pyodbc.connect(connection_string)
        logger.info("Connected to SQL Server")
    
    def load_mmm_data(
        self,
        *,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        if not self.connection:
            self.connect()
        
        # Build query
        if query:
            sql = query
        else:
            if not table:
                raise ValueError("Must provide either query or table")
            
            # Use schema if provided
            table_name = f"{schema}.{table}" if schema else table
            sql = f"SELECT * FROM {table_name}"
            
            if filters:
                conditions = []
                for col, values in filters.items():
                    if isinstance(values, list):
                        values_str = ', '.join(f"'{v}'" for v in values)
                        conditions.append(f"{col} IN ({values_str})")
                    else:
                        conditions.append(f"{col} = '{values}'")
                sql += " WHERE " + " AND ".join(conditions)
        
        df = pd.read_sql(sql, self.connection)
        logger.info(f"Loaded {len(df)} rows from SQL Server")
        
        return df


class OracleConnector(DatabaseConnector):
    """Oracle Database connector"""
    
    def connect(self) -> None:
        try:
            import oracledb
        except ImportError:
            raise ImportError("oracledb not installed. Install with: pip install oracledb")
        
        # Parse connection URL
        # Format: oracle://user:pass@host:port/service_name
        import urllib.parse
        parsed = urllib.parse.urlparse(self.connection_url)
        
        self.connection = oracledb.connect(
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port or 1521,
            service_name=parsed.path.lstrip('/')
        )
        logger.info("Connected to Oracle")
    
    def load_mmm_data(
        self,
        *,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        if not self.connection:
            self.connect()
        
        # Build query
        if query:
            sql = query
        else:
            if not table:
                raise ValueError("Must provide either query or table")
            
            # Use schema if provided
            table_name = f"{schema}.{table}" if schema else table
            sql = f"SELECT * FROM {table_name}"
            
            if filters:
                conditions = []
                for col, values in filters.items():
                    if isinstance(values, list):
                        values_str = ', '.join(f"'{v}'" for v in values)
                        conditions.append(f"{col} IN ({values_str})")
                    else:
                        conditions.append(f"{col} = '{values}'")
                sql += " WHERE " + " AND ".join(conditions)
        
        df = pd.read_sql(sql, self.connection)
        logger.info(f"Loaded {len(df)} rows from Oracle")
        
        return df


class RedshiftConnector(DatabaseConnector):
    """Amazon Redshift connector"""
    
    def connect(self) -> None:
        try:
            import psycopg2
        except ImportError:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
        
        # Redshift uses PostgreSQL protocol
        self.connection = psycopg2.connect(self.connection_url)
        logger.info("Connected to Redshift")
    
    def load_mmm_data(
        self,
        *,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        if not self.connection:
            self.connect()
        
        # Build query
        if query:
            sql = query
        else:
            if not schema or not table:
                raise ValueError("Must provide either query or (schema + table)")
            sql = f"SELECT * FROM {schema}.{table}"
            
            if filters:
                conditions = []
                for col, values in filters.items():
                    if isinstance(values, list):
                        values_str = ', '.join(f"'{v}'" for v in values)
                        conditions.append(f"{col} IN ({values_str})")
                    else:
                        conditions.append(f"{col} = '{values}'")
                sql += " WHERE " + " AND ".join(conditions)
        
        logger.info(f"Executing Redshift query: {sql[:100]}...")
        df = pd.read_sql(sql, self.connection)
        logger.info(f"Loaded {len(df)} rows from Redshift")
        
        return df


class DuckDBConnector(DatabaseConnector):
    """DuckDB connector (analytical database)"""
    
    def connect(self) -> None:
        try:
            import duckdb
        except ImportError:
            raise ImportError("duckdb not installed. Install with: pip install duckdb")
        
        # Extract path from URL (duckdb:///path/to/db.duckdb or duckdb://:memory:)
        db_path = self.connection_url.replace('duckdb:///', '').replace('duckdb://', '')
        
        if db_path == ':memory:' or db_path == '':
            self.connection = duckdb.connect(':memory:')
            logger.info("Connected to DuckDB (in-memory)")
        else:
            self.connection = duckdb.connect(db_path)
            logger.info(f"Connected to DuckDB: {db_path}")
    
    def load_mmm_data(
        self,
        *,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        if not self.connection:
            self.connect()
        
        # Build query
        if query:
            sql = query
        else:
            if not table:
                raise ValueError("Must provide either query or table")
            
            # Use schema if provided
            table_name = f"{schema}.{table}" if schema else table
            sql = f"SELECT * FROM {table_name}"
            
            if filters:
                conditions = []
                for col, values in filters.items():
                    if isinstance(values, list):
                        values_str = ', '.join(f"'{v}'" for v in values)
                        conditions.append(f"{col} IN ({values_str})")
                    else:
                        conditions.append(f"{col} = '{values}'")
                sql += " WHERE " + " AND ".join(conditions)
        
        df = self.connection.execute(sql).fetchdf()
        logger.info(f"Loaded {len(df)} rows from DuckDB")
        
        return df


class ParquetConnector(DatabaseConnector):
    """Parquet file connector (read Parquet files as if they were a database)"""
    
    def __init__(self, file_path: str):
        """
        Parameters
        ----------
        file_path : str
            Path to Parquet file or directory containing Parquet files
        """
        self.file_path = file_path
        self.connection = None
        logger.info(f"Initialized ParquetConnector: {file_path}")
    
    def connect(self) -> None:
        # No connection needed for Parquet files
        logger.info("Parquet files don't require connection")
    
    def load_mmm_data(
        self,
        *,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Load data from Parquet file(s).
        
        Parameters
        ----------
        schema : str, optional
            Not used for Parquet (included for API consistency)
        table : str, optional
            Not used for Parquet (included for API consistency)
        query : str, optional
            Not used for Parquet (included for API consistency)
        filters : Dict, optional
            Column filters to apply after loading
        """
        import pyarrow.parquet as pq
        from pathlib import Path
        
        file_path = Path(self.file_path)
        
        # Read Parquet file(s)
        if file_path.is_file():
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} rows from Parquet file: {file_path}")
        elif file_path.is_dir():
            # Read all Parquet files in directory
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} rows from Parquet directory: {file_path}")
        else:
            raise FileNotFoundError(f"Parquet file/directory not found: {file_path}")
        
        # Apply filters if provided
        if filters:
            for col, values in filters.items():
                if isinstance(values, list):
                    df = df[df[col].isin(values)]
                else:
                    df = df[df[col] == values]
            logger.info(f"Applied filters, {len(df)} rows remaining")
        
        return df

