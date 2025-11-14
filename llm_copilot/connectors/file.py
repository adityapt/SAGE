"""
File Connectors

Load MMM data from file sources (CSV, Excel, Google Sheets, JSON).
"""

from typing import Optional, Dict
from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FileConnector:
    """
    Base class for file connectors.
    
    Supports:
    - CSV
    - Excel (XLS, XLSX)
    - Google Sheets
    - JSON
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        logger.info(f"Initialized FileConnector: {file_path}")
    
    @classmethod
    def from_path(cls, file_path: str) -> 'FileConnector':
        """Create connector based on file extension"""
        path = Path(file_path)
        
        if path.suffix.lower() in ['.csv', '.tsv', '.txt']:
            return CSVConnector(file_path)
        elif path.suffix.lower() in ['.xls', '.xlsx', '.xlsm']:
            return ExcelConnector(file_path)
        elif path.suffix.lower() == '.json':
            return JSONConnector(file_path)
        elif 'docs.google.com/spreadsheets' in file_path:
            return GoogleSheetsConnector(file_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def load_mmm_data(self, filters: Optional[Dict] = None) -> pd.DataFrame:
        """Load MMM data from file"""
        raise NotImplementedError


class CSVConnector(FileConnector):
    """CSV file connector"""
    
    def load_mmm_data(
        self, 
        *,
        filters: Optional[Dict] = None,
        sep: str = ',',
        encoding: str = 'utf-8'
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Parameters
        ----------
        filters : Dict, optional
            Column filters to apply after loading
        sep : str, default=','
            Delimiter (use '\t' for TSV)
        encoding : str, default='utf-8'
            File encoding
            
        Returns
        -------
        pd.DataFrame
            MMM data
        """
        logger.info(f"Loading CSV: {self.file_path}")
        
        df = pd.read_csv(self.file_path, sep=sep, encoding=encoding)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Apply filters if provided
        if filters:
            for col, values in filters.items():
                if isinstance(values, list):
                    df = df[df[col].isin(values)]
                else:
                    df = df[df[col] == values]
            logger.info(f"Applied filters, {len(df)} rows remaining")
        
        return df


class ExcelConnector(FileConnector):
    """Excel file connector"""
    
    def load_mmm_data(
        self,
        *,
        sheet_name: str = 0,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Parameters
        ----------
        sheet_name : str or int, default=0
            Sheet name or index (0 = first sheet)
        filters : Dict, optional
            Column filters to apply after loading
            
        Returns
        -------
        pd.DataFrame
            MMM data
        """
        logger.info(f"Loading Excel: {self.file_path}, sheet: {sheet_name}")
        
        df = pd.read_excel(self.file_path, sheet_name=sheet_name)
        logger.info(f"Loaded {len(df)} rows from Excel")
        
        # Apply filters if provided
        if filters:
            for col, values in filters.items():
                if isinstance(values, list):
                    df = df[df[col].isin(values)]
                else:
                    df = df[df[col] == values]
            logger.info(f"Applied filters, {len(df)} rows remaining")
        
        return df


class JSONConnector(FileConnector):
    """JSON file connector"""
    
    def load_mmm_data(
        self,
        *,
        orient: str = 'records',
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Load data from JSON file.
        
        Parameters
        ----------
        orient : str, default='records'
            JSON orientation ('records', 'index', 'columns', 'values')
        filters : Dict, optional
            Column filters to apply after loading
            
        Returns
        -------
        pd.DataFrame
            MMM data
        """
        logger.info(f"Loading JSON: {self.file_path}")
        
        df = pd.read_json(self.file_path, orient=orient)
        logger.info(f"Loaded {len(df)} rows from JSON")
        
        # Apply filters if provided
        if filters:
            for col, values in filters.items():
                if isinstance(values, list):
                    df = df[df[col].isin(values)]
                else:
                    df = df[df[col] == values]
            logger.info(f"Applied filters, {len(df)} rows remaining")
        
        return df


class GoogleSheetsConnector(FileConnector):
    """Google Sheets connector"""
    
    def load_mmm_data(
        self,
        *,
        sheet_name: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Load data from Google Sheets.
        
        Parameters
        ----------
        sheet_name : str, optional
            Sheet name (tab) to load (default: first sheet)
        filters : Dict, optional
            Column filters to apply after loading
            
        Returns
        -------
        pd.DataFrame
            MMM data
            
        Notes
        -----
        Requires the sheet to be publicly accessible or authenticated via
        Google API credentials.
        """
        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except ImportError:
            raise ImportError(
                "gspread and google-auth not installed. "
                "Install with: pip install gspread google-auth"
            )
        
        logger.info(f"Loading Google Sheets: {self.file_path}")
        
        # Extract sheet ID from URL
        # Format: https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit...
        sheet_id = self._extract_sheet_id(self.file_path)
        
        # Try to load without authentication (public sheets)
        try:
            url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            if sheet_name:
                url += f"&gid={sheet_name}"
            
            df = pd.read_csv(url)
            logger.info(f"Loaded {len(df)} rows from Google Sheets (public)")
            
        except Exception as e:
            # Try with authentication
            logger.warning(f"Public access failed: {e}. Trying with authentication...")
            df = self._load_with_auth(sheet_id, sheet_name)
        
        # Apply filters if provided
        if filters:
            for col, values in filters.items():
                if isinstance(values, list):
                    df = df[df[col].isin(values)]
                else:
                    df = df[df[col] == values]
            logger.info(f"Applied filters, {len(df)} rows remaining")
        
        return df
    
    def _extract_sheet_id(self, url: str) -> str:
        """Extract sheet ID from Google Sheets URL"""
        import re
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url)
        if not match:
            raise ValueError(f"Invalid Google Sheets URL: {url}")
        return match.group(1)
    
    def _load_with_auth(self, sheet_id: str, sheet_name: Optional[str]) -> pd.DataFrame:
        """Load with Google API authentication"""
        import gspread
        from google.oauth2.service_account import Credentials
        import os
        
        # Look for credentials file
        creds_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not creds_file:
            raise ValueError(
                "Google authentication required. "
                "Set GOOGLE_APPLICATION_CREDENTIALS environment variable "
                "to path of service account JSON file"
            )
        
        # Authenticate
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly'
        ]
        creds = Credentials.from_service_account_file(creds_file, scopes=scopes)
        client = gspread.authorize(creds)
        
        # Open sheet
        sheet = client.open_by_key(sheet_id)
        
        # Get worksheet
        if sheet_name:
            worksheet = sheet.worksheet(sheet_name)
        else:
            worksheet = sheet.sheet1  # First sheet
        
        # Convert to DataFrame
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        logger.info(f"Loaded {len(df)} rows from Google Sheets (authenticated)")
        return df


class ParquetFileConnector(FileConnector):
    """Parquet file connector (convenience wrapper)"""
    
    def load_mmm_data(
        self,
        *,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Load data from Parquet file.
        
        Parameters
        ----------
        filters : Dict, optional
            Column filters to apply after loading
            
        Returns
        -------
        pd.DataFrame
            MMM data
        """
        logger.info(f"Loading Parquet: {self.file_path}")
        
        df = pd.read_parquet(self.file_path)
        logger.info(f"Loaded {len(df)} rows from Parquet")
        
        # Apply filters if provided
        if filters:
            for col, values in filters.items():
                if isinstance(values, list):
                    df = df[df[col].isin(values)]
                else:
                    df = df[df[col] == values]
            logger.info(f"Applied filters, {len(df)} rows remaining")
        
        return df



