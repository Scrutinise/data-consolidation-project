#!/usr/bin/env python3
"""
File Structure Analyzer for Trading Data
Processes each file individually and documents its structure in SQLite database
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Try importing PDF libraries
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  pdfplumber not installed - PDF analysis will be limited")

# Try importing xlsb library
try:
    import pyxlsb
    XLSB_AVAILABLE = True
except ImportError:
    XLSB_AVAILABLE = False
    print("⚠️  pyxlsb not installed - XLSB analysis will be limited")


class FileAnalyzer:
    """Analyzes trading data files one at a time"""
    
    def __init__(self, raw_data_path, db_path):
        self.raw_data_path = Path(raw_data_path)
        self.db_path = Path(db_path)
        self.conn = None
        self.files_processed = 0
        
    def create_database(self):
        """Create SQLite database with metadata table"""
        print(f"📁 Creating database: {self.db_path}")
        
        # Create parent directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT,
                columns_found TEXT,
                row_count INTEGER,
                date_range_start TEXT,
                date_range_end TEXT,
                account_id TEXT,
                file_size_kb REAL,
                analysis_timestamp TEXT,
                notes TEXT
            )
        """)
        
        self.conn.commit()
        print("✅ Database created successfully\n")
        
    def analyze_pdf(self, file_path):
        """Analyze PDF file structure"""
        file_name = file_path.name
        file_size = file_path.stat().st_size / 1024  # KB
        
        metadata = {
            'filename': file_name,
            'file_type': 'PDF_daily' if 'day' in file_name.lower() else 'PDF_monthly',
            'file_size_kb': round(file_size, 2),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        if not PDF_AVAILABLE:
            metadata['notes'] = 'pdfplumber not available - could not extract structure'
            metadata['columns_found'] = 'Unknown'
            metadata['row_count'] = 0
            return metadata
            
        try:
            with pdfplumber.open(file_path) as pdf:
                # Get basic info
                num_pages = len(pdf.pages)
                
                # Try to extract tables from first page
                first_page = pdf.pages[0]
                tables = first_page.extract_tables()
                
                # Try to find transaction table
                columns = []
                row_count = 0
                account_id = None
                
                # Look for account ID in text
                text = first_page.extract_text()
                if 'Account ID:' in text:
                    for line in text.split('\n'):
                        if 'Account ID:' in line:
                            account_id = line.split('Account ID:')[-1].strip().split()[0]
                            break
                
                # Analyze tables
                for table in tables:
                    if table and len(table) > 0:
                        # First row might be headers
                        headers = table[0]
                        if headers and any(h for h in headers if h):  # Has non-empty values
                            # Clean up headers
                            clean_headers = [str(h).strip() for h in headers if h]
                            if clean_headers:
                                columns = clean_headers
                                row_count = len(table) - 1  # Exclude header row
                                break
                
                metadata['columns_found'] = ', '.join(columns) if columns else 'No clear table structure'
                metadata['row_count'] = row_count
                metadata['account_id'] = account_id
                metadata['notes'] = f'{num_pages} pages'
                
        except Exception as e:
            metadata['notes'] = f'Error: {str(e)}'
            metadata['columns_found'] = 'Error during extraction'
            metadata['row_count'] = 0
            
        return metadata
    
    def analyze_xlsx(self, file_path):
        """Analyze XLSX file structure"""
        file_name = file_path.name
        file_size = file_path.stat().st_size / 1024  # KB
        
        metadata = {
            'filename': file_name,
            'file_type': 'XLSX',
            'file_size_kb': round(file_size, 2),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Read Excel file - just get basic info without loading all data
            xl_file = pd.ExcelFile(file_path, engine='openpyxl')
            sheet_names = xl_file.sheet_names
            
            # Read first sheet, just first few rows to get structure
            df = pd.read_excel(file_path, sheet_name=0, nrows=5)
            
            columns = list(df.columns)
            
            # Now get actual row count (without loading all data into memory)
            df_full = pd.read_excel(file_path, sheet_name=0)
            row_count = len(df_full)
            
            # Try to find date columns and get range
            date_cols = [col for col in df_full.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                try:
                    dates = pd.to_datetime(df_full[date_cols[0]], errors='coerce')
                    metadata['date_range_start'] = str(dates.min())
                    metadata['date_range_end'] = str(dates.max())
                except:
                    pass
            
            # Look for account ID column
            account_cols = [col for col in df_full.columns if 'account' in col.lower()]
            if account_cols:
                account_values = df_full[account_cols[0]].dropna().unique()
                if len(account_values) > 0:
                    metadata['account_id'] = str(account_values[0])
            
            metadata['columns_found'] = ', '.join([str(col) for col in columns])
            metadata['row_count'] = row_count
            metadata['notes'] = f'Sheets: {", ".join(sheet_names)}'
            
        except Exception as e:
            metadata['notes'] = f'Error: {str(e)}'
            metadata['columns_found'] = 'Error during extraction'
            metadata['row_count'] = 0
            
        return metadata
    
    def analyze_xlsb(self, file_path):
        """Analyze XLSB (binary Excel) file structure"""
        file_name = file_path.name
        file_size = file_path.stat().st_size / 1024  # KB
        
        metadata = {
            'filename': file_name,
            'file_type': 'XLSB',
            'file_size_kb': round(file_size, 2),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        if not XLSB_AVAILABLE:
            metadata['notes'] = 'pyxlsb not available - trying pandas with pyxlsb engine'
            # Try with pandas anyway
            try:
                df = pd.read_excel(file_path, engine='pyxlsb', nrows=5)
                columns = list(df.columns)
                
                df_full = pd.read_excel(file_path, engine='pyxlsb')
                row_count = len(df_full)
                
                metadata['columns_found'] = ', '.join([str(col) for col in columns])
                metadata['row_count'] = row_count
                return metadata
            except Exception as e:
                metadata['notes'] = f'Error: {str(e)}'
                metadata['columns_found'] = 'Error - pyxlsb required'
                metadata['row_count'] = 0
                return metadata
        
        try:
            # Use pandas with pyxlsb engine
            df = pd.read_excel(file_path, engine='pyxlsb', nrows=5)
            columns = list(df.columns)
            
            df_full = pd.read_excel(file_path, engine='pyxlsb')
            row_count = len(df_full)
            
            # Try to find date range
            date_cols = [col for col in df_full.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                try:
                    dates = pd.to_datetime(df_full[date_cols[0]], errors='coerce')
                    metadata['date_range_start'] = str(dates.min())
                    metadata['date_range_end'] = str(dates.max())
                except:
                    pass
            
            # Look for account ID
            account_cols = [col for col in df_full.columns if 'account' in col.lower()]
            if account_cols:
                account_values = df_full[account_cols[0]].dropna().unique()
                if len(account_values) > 0:
                    metadata['account_id'] = str(account_values[0])
            
            metadata['columns_found'] = ', '.join([str(col) for col in columns])
            metadata['row_count'] = row_count
            
        except Exception as e:
            metadata['notes'] = f'Error: {str(e)}'
            metadata['columns_found'] = 'Error during extraction'
            metadata['row_count'] = 0
            
        return metadata
    
    def save_metadata(self, metadata):
        """Save metadata to database"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO file_analysis 
            (filename, file_type, columns_found, row_count, date_range_start, 
             date_range_end, account_id, file_size_kb, analysis_timestamp, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.get('filename'),
            metadata.get('file_type'),
            metadata.get('columns_found'),
            metadata.get('row_count'),
            metadata.get('date_range_start'),
            metadata.get('date_range_end'),
            metadata.get('account_id'),
            metadata.get('file_size_kb'),
            metadata.get('analysis_timestamp'),
            metadata.get('notes')
        ))
        
        self.conn.commit()
    
    def process_all_files(self):
        """Process all files in raw data directory"""
        print(f"🔍 Scanning directory: {self.raw_data_path}\n")
        
        # Get all files
        all_files = sorted(self.raw_data_path.glob('*'))
        
        # Filter to only data files
        data_files = [f for f in all_files if f.suffix.lower() in ['.pdf', '.xlsx', '.xlsb']]
        
        total_files = len(data_files)
        print(f"📊 Found {total_files} data files to analyze\n")
        print("="*70)
        
        for idx, file_path in enumerate(data_files, 1):
            print(f"\n[{idx}/{total_files}] Analyzing: {file_path.name}")
            
            # Determine file type and process accordingly
            suffix = file_path.suffix.lower()
            
            try:
                if suffix == '.pdf':
                    metadata = self.analyze_pdf(file_path)
                elif suffix == '.xlsx':
                    metadata = self.analyze_xlsx(file_path)
                elif suffix == '.xlsb':
                    metadata = self.analyze_xlsb(file_path)
                else:
                    print(f"⚠️  Skipping unknown file type: {suffix}")
                    continue
                
                # Save to database
                self.save_metadata(metadata)
                self.files_processed += 1
                
                # Show summary
                print(f"   Type: {metadata.get('file_type')}")
                print(f"   Rows: {metadata.get('row_count', 0)}")
                if metadata.get('account_id'):
                    print(f"   Account: {metadata.get('account_id')}")
                if metadata.get('notes'):
                    print(f"   Notes: {metadata.get('notes')}")
                    
                # Progress marker every 5 files
                if idx % 5 == 0:
                    print(f"\n{'='*70}")
                    print(f"✅ Progress: {idx}/{total_files} files analyzed")
                    print(f"{'='*70}")
                    
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {str(e)}")
                continue
        
        print(f"\n{'='*70}")
        print(f"🎉 Analysis Complete!")
        print(f"{'='*70}")
        print(f"✅ Successfully analyzed: {self.files_processed}/{total_files} files")
        print(f"📁 Results saved to: {self.db_path}")
    
    def export_summary(self):
        """Export metadata to CSV and create summary report"""
        print(f"\n📊 Exporting summary reports...")
        
        # Export to CSV
        csv_path = self.db_path.parent / 'file_metadata.csv'
        df = pd.read_sql_query("SELECT * FROM file_analysis", self.conn)
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV exported: {csv_path}")
        
        # Create summary text file
        summary_path = self.db_path.parent / 'schema_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FILE STRUCTURE ANALYSIS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            # Group by file type
            for file_type in df['file_type'].unique():
                if pd.isna(file_type):
                    continue
                    
                type_df = df[df['file_type'] == file_type]
                f.write(f"\n{file_type} FILES ({len(type_df)} files)\n")
                f.write("-" * 70 + "\n")
                
                # Get unique columns across all files of this type
                all_columns = set()
                for cols in type_df['columns_found'].dropna():
                    if cols != 'Unknown' and 'Error' not in cols:
                        all_columns.update([c.strip() for c in str(cols).split(',')])
                
                if all_columns:
                    f.write("Common columns found:\n")
                    for col in sorted(all_columns):
                        f.write(f"  - {col}\n")
                
                # Total rows
                total_rows = type_df['row_count'].sum()
                f.write(f"\nTotal transactions: {total_rows:,}\n")
                
                # Date range if available
                dates_start = type_df['date_range_start'].dropna()
                dates_end = type_df['date_range_end'].dropna()
                if len(dates_start) > 0:
                    f.write(f"Date range: {dates_start.min()} to {dates_end.max()}\n")
                
                # Account IDs
                accounts = type_df['account_id'].dropna().unique()
                if len(accounts) > 0:
                    f.write(f"Account IDs: {', '.join(accounts)}\n")
                
                f.write("\n")
        
        print(f"✅ Summary exported: {summary_path}")
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("TRADING DATA FILE STRUCTURE ANALYZER")
    print("="*70 + "\n")
    
    # Set paths
    raw_data_path = Path("data/raw")
    db_path = Path("data/file_metadata.db")
    
    # Check if raw data directory exists
    if not raw_data_path.exists():
        print(f"❌ Error: Directory not found: {raw_data_path}")
        print(f"   Current working directory: {Path.cwd()}")
        sys.exit(1)
    
    # Create analyzer
    analyzer = FileAnalyzer(raw_data_path, db_path)
    
    try:
        # Create database
        analyzer.create_database()
        
        # Process all files
        analyzer.process_all_files()
        
        # Export summary
        analyzer.export_summary()
        
    finally:
        analyzer.close()
    
    print(f"\n✅ All done! Check the following files:")
    print(f"   - {db_path} (SQLite database)")
    print(f"   - {db_path.parent}/file_metadata.csv (CSV export)")
    print(f"   - {db_path.parent}/schema_summary.txt (Text summary)")
    print()


if __name__ == "__main__":
    main()
