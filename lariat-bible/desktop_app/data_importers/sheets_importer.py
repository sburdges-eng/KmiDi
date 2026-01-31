"""
Google Sheets Data Importer
Import data from Google Sheets into the application
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleSheetsImporter:
    """Import and process Google Sheets data"""

    def __init__(self, credentials_file: str = None):
        """
        Initialize Google Sheets Importer

        Args:
            credentials_file: Path to Google API credentials JSON file
        """
        self.credentials_file = credentials_file
        self.imported_data = {}
        self.api_key = None

        # Try to load API key from environment or credentials file
        self._load_credentials()

    def _load_credentials(self):
        """Load Google API credentials"""
        # Try environment variable first
        self.api_key = os.getenv('GOOGLE_SHEETS_API_KEY')

        if not self.api_key and self.credentials_file:
            try:
                with open(self.credentials_file, 'r') as f:
                    creds = json.load(f)
                    self.api_key = creds.get('api_key')
            except Exception as e:
                logger.warning(f"Could not load credentials: {e}")

    def import_sheet(self, spreadsheet_id: str, range_name: str = 'Sheet1!A:Z',
                    has_header: bool = True) -> Optional[List[Dict]]:
        """
        Import data from Google Sheet

        Args:
            spreadsheet_id: Google Sheets ID (from URL)
            range_name: Range to import (e.g., 'Sheet1!A:Z')
            has_header: Whether first row contains headers

        Returns:
            List of dictionaries representing rows
        """
        if not self.api_key:
            logger.error("No API key found. Set GOOGLE_SHEETS_API_KEY environment variable or provide credentials file")
            return None

        try:
            url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}"
            params = {'key': self.api_key}

            response = requests.get(url, params=params)
            response.raise_for_status()

            result = response.json()
            values = result.get('values', [])

            if not values:
                logger.warning("No data found in sheet")
                return []

            data = []
            if has_header:
                headers = values[0]
                for row in values[1:]:
                    if row:  # Skip empty rows
                        # Pad row with empty strings if shorter than headers
                        row = row + [''] * (len(headers) - len(row))
                        row_dict = {headers[i]: row[i] if i < len(row) else ''
                                  for i in range(len(headers))}
                        data.append(row_dict)
            else:
                for row in values:
                    if row:
                        row_dict = {f'col_{i}': val for i, val in enumerate(row)}
                        data.append(row_dict)

            key = f"{spreadsheet_id}:{range_name}"
            self.imported_data[key] = data
            logger.info(f"Imported {len(data)} rows from Google Sheet")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error accessing Google Sheets: {e}")
            return None
        except Exception as e:
            logger.error(f"Error importing sheet: {e}")
            return None

    def import_sheet_by_url(self, url: str, range_name: str = 'Sheet1!A:Z',
                           has_header: bool = True) -> Optional[List[Dict]]:
        """
        Import data from Google Sheet using full URL

        Args:
            url: Full Google Sheets URL
            range_name: Range to import
            has_header: Whether first row contains headers

        Returns:
            List of dictionaries representing rows
        """
        # Extract spreadsheet ID from URL
        spreadsheet_id = self._extract_spreadsheet_id(url)
        if not spreadsheet_id:
            logger.error("Could not extract spreadsheet ID from URL")
            return None

        return self.import_sheet(spreadsheet_id, range_name, has_header)

    def _extract_spreadsheet_id(self, url: str) -> Optional[str]:
        """
        Extract spreadsheet ID from Google Sheets URL

        Args:
            url: Google Sheets URL

        Returns:
            Spreadsheet ID or None
        """
        # Handle different URL formats
        if '/d/' in url:
            start = url.find('/d/') + 3
            end = url.find('/', start)
            if end == -1:
                end = url.find('?', start)
            if end == -1:
                end = len(url)
            return url[start:end]
        return None

    def import_public_sheet(self, spreadsheet_id: str, range_name: str = 'Sheet1!A:Z',
                           has_header: bool = True) -> Optional[List[Dict]]:
        """
        Import data from a public Google Sheet (no API key required)

        Args:
            spreadsheet_id: Google Sheets ID
            range_name: Range to import
            has_header: Whether first row contains headers

        Returns:
            List of dictionaries representing rows
        """
        try:
            # Use CSV export for public sheets
            url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv"

            response = requests.get(url)
            response.raise_for_status()

            # Parse CSV data
            import csv
            import io

            data = []
            csv_data = io.StringIO(response.text)
            reader = csv.reader(csv_data)

            rows = list(reader)
            if not rows:
                return []

            if has_header:
                headers = rows[0]
                for row in rows[1:]:
                    if row:
                        row = row + [''] * (len(headers) - len(row))
                        row_dict = {headers[i]: row[i] if i < len(row) else ''
                                  for i in range(len(headers))}
                        data.append(row_dict)
            else:
                for row in rows:
                    if row:
                        row_dict = {f'col_{i}': val for i, val in enumerate(row)}
                        data.append(row_dict)

            key = f"{spreadsheet_id}:public"
            self.imported_data[key] = data
            logger.info(f"Imported {len(data)} rows from public Google Sheet")
            return data

        except Exception as e:
            logger.error(f"Error importing public sheet: {e}")
            return None

    def get_sheet_properties(self, spreadsheet_id: str) -> Optional[Dict]:
        """
        Get properties of a Google Sheet

        Args:
            spreadsheet_id: Google Sheets ID

        Returns:
            Dictionary with sheet properties
        """
        if not self.api_key:
            logger.error("No API key found")
            return None

        try:
            url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}"
            params = {'key': self.api_key}

            response = requests.get(url, params=params)
            response.raise_for_status()

            result = response.json()

            properties = {
                'title': result.get('properties', {}).get('title'),
                'sheets': []
            }

            for sheet in result.get('sheets', []):
                sheet_props = sheet.get('properties', {})
                properties['sheets'].append({
                    'title': sheet_props.get('title'),
                    'sheet_id': sheet_props.get('sheetId'),
                    'index': sheet_props.get('index'),
                    'row_count': sheet_props.get('gridProperties', {}).get('rowCount'),
                    'column_count': sheet_props.get('gridProperties', {}).get('columnCount')
                })

            return properties

        except Exception as e:
            logger.error(f"Error getting sheet properties: {e}")
            return None

    def import_menu_from_sheets(self, spreadsheet_id: str, sheet_name: str = 'Menu') -> Optional[Dict]:
        """
        Import menu data from Google Sheets

        Args:
            spreadsheet_id: Google Sheets ID
            sheet_name: Name of the sheet containing menu data

        Returns:
            Dictionary with menu data
        """
        range_name = f"{sheet_name}!A:Z"
        data = self.import_sheet(spreadsheet_id, range_name)

        if not data:
            return None

        menu_data = {
            'items': [],
            'categories': set()
        }

        for row in data:
            item = {
                'name': row.get('Item') or row.get('Name'),
                'category': row.get('Category'),
                'cost': self._safe_float(row.get('Cost')),
                'price': self._safe_float(row.get('Price')),
                'description': row.get('Description')
            }

            if item['name']:
                menu_data['items'].append(item)
                if item['category']:
                    menu_data['categories'].add(item['category'])

        menu_data['categories'] = list(menu_data['categories'])
        return menu_data

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float"""
        try:
            if value == '' or value is None:
                return default
            # Remove currency symbols and commas
            if isinstance(value, str):
                value = value.replace('$', '').replace(',', '').strip()
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_summary(self) -> Dict:
        """
        Get summary of imported data

        Returns:
            Summary dictionary
        """
        return {
            'sheets_imported': len(self.imported_data),
            'total_rows': sum(len(data) for data in self.imported_data.values()),
            'sheets': list(self.imported_data.keys())
        }

    @staticmethod
    def get_setup_instructions() -> str:
        """
        Get instructions for setting up Google Sheets API access

        Returns:
            Setup instructions
        """
        return """
        Google Sheets Import Setup Instructions:

        1. Go to Google Cloud Console: https://console.cloud.google.com
        2. Create a new project or select existing
        3. Enable Google Sheets API
        4. Create an API key (Credentials > Create Credentials > API Key)
        5. Set the API key as environment variable:
           export GOOGLE_SHEETS_API_KEY='your-api-key-here'

        Or provide a credentials JSON file with:
        {
            "api_key": "your-api-key-here"
        }

        For public sheets, no API key is required!
        Use import_public_sheet() method instead.
        """


if __name__ == "__main__":
    # Test the importer
    importer = GoogleSheetsImporter()

    print("Google Sheets Importer initialized")
    print("\nSetup instructions:")
    print(GoogleSheetsImporter.get_setup_instructions())
