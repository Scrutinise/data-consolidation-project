#!/usr/bin/env python3
"""
Incremental Data Addition Script
Adds missing original Excel files to existing consolidated dataset
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class IncrementalDataAdder:
    """Adds missing Excel files to existing consolidated dataset"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.new_trades = []
        
    def parse_date(self, date_value):
        """Parse various date formats into standardized format"""
        if pd.isna(date_value):
            return None
        
        # Handle Excel datetime
        if isinstance(date_value, datetime):
            return date_value
        
        # Handle timestamp (epoch issue in XLSB files)
        if isinstance(date_value, (int, float)):
            # Check if it's a reasonable timestamp
            if date_value > 1000000000 and date_value < 2000000000:
                # Looks like Unix timestamp
                return datetime.fromtimestamp(date_value)
            elif date_value < 100000:
                # Likely Excel serial date
                return pd.Timestamp('1899-12-30') + pd.Timedelta(days=date_value)
            else:
                # Might be microseconds - try different units
                try:
                    # Try microseconds
                    dt = pd.Timestamp(date_value, unit='us')
                    # Check if result is reasonable (between 1990 and 2030)
                    if dt.year >= 1990 and dt.year <= 2030:
                        return dt
                    # Try milliseconds
                    dt = pd.Timestamp(date_value, unit='ms')
                    if dt.year >= 1990 and dt.year <= 2030:
                        return dt
                    # Try seconds
                    dt = pd.Timestamp(date_value, unit='s')
                    if dt.year >= 1990 and dt.year <= 2030:
                        return dt
                except:
                    pass
                return None
        
        # Handle string dates
        if isinstance(date_value, str):
            try:
                # Try various formats
                for fmt in ['%d.%m.%y %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S']:
                    try:
                        return datetime.strptime(date_value, fmt)
                    except:
                        continue
                
                # Try pandas parser
                return pd.to_datetime(date_value)
            except:
                return None
        
        return None
    
    def extract_original_xlsx(self, file_path):
        """Extract data from original XLSX files (Schema A format)"""
        print(f"  Reading: {file_path.name}")
        
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            
            print(f"    Columns found: {list(df.columns)[:5]}...")
            print(f"    Total rows: {len(df)}")
            
            # Schema A: ACCOUNT ID, BOOKING TIME, Booking type, AMOUNT CCY, RELATED INSTRUMENT, AMOUNT, C/F BALANCE
            trades = []
            
            for idx, row in df.iterrows():
                trade = {
                    'transaction_time': row.get('BOOKING TIME'),
                    'type': row.get('Booking type'),
                    'order_id': None,  # Not in this schema
                    'trade_id': None,  # Not in this schema
                    'correlated_order_id': None,
                    'product': row.get('RELATED INSTRUMENT'),
                    'amt_per_point': None,  # Not in this schema
                    'trade_price': None,  # Not in this schema
                    'trade_value': row.get('AMOUNT'),
                    'realised_pnl': None,  # Different concept in this schema
                    'amount_ccy': row.get('AMOUNT CCY'),
                    'cf_balance': row.get('C/F BALANCE'),
                    'account_id': row.get('ACCOUNT ID'),
                    'source_file': file_path.name,
                    'source_type': 'original_xlsx'
                }
                trades.append(trade)
            
            print(f"    Extracted: {len(trades)} transactions")
            return trades
            
        except Exception as e:
            print(f"  ERROR reading {file_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_original_xlsb(self, file_path):
        """Extract data from original XLSB files (Schema B format)"""
        print(f"  Reading: {file_path.name}")
        
        try:
            df = pd.read_excel(file_path, engine='pyxlsb')
            
            print(f"    Columns found: {list(df.columns)[:5]}...")
            print(f"    Total rows: {len(df)}")
            
            trades = []
            
            # Determine which schema based on columns
            if 'Trade Time' in df.columns:
                # Schema B1: Trade History format
                print(f"    Schema: Trade History format")
                for idx, row in df.iterrows():
                    trade = {
                        'transaction_time': row.get('Trade Time'),
                        'type': row.get('Direction'),  # Buy/Sell
                        'order_id': row.get('Order ID'),
                        'trade_id': row.get('Trade ID'),
                        'correlated_order_id': None,
                        'product': row.get('Instrument'),
                        'amt_per_point': row.get('Point Value'),
                        'trade_price': row.get('Price'),
                        'trade_value': None,
                        'realised_pnl': None,
                        'quantity': row.get('Quantity'),
                        'account_id': row.get('Account ID'),
                        'source_file': file_path.name,
                        'source_type': 'original_xlsb'
                    }
                    trades.append(trade)
            
            elif 'Time' in df.columns and 'Transaction Type' in df.columns:
                # Schema B2: Trade and Cash History format
                print(f"    Schema: Trade and Cash History format")
                for idx, row in df.iterrows():
                    trade = {
                        'transaction_time': row.get('Time'),
                        'type': row.get('Transaction Type'),
                        'order_id': None,
                        'trade_id': row.get('Transaction ID'),
                        'correlated_order_id': None,
                        'product': None,
                        'amt_per_point': None,
                        'trade_price': None,
                        'trade_value': row.get('QTY'),  # Might be quantity or amount
                        'realised_pnl': None,
                        'account_id': row.get('Account ID'),
                        'source_file': file_path.name,
                        'source_type': 'original_xlsb'
                    }
                    trades.append(trade)
            else:
                print(f"    WARNING: Unknown schema in {file_path.name}")
                print(f"    Columns: {list(df.columns)}")
            
            print(f"    Extracted: {len(trades)} transactions")
            return trades
            
        except Exception as e:
            print(f"  ERROR reading {file_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def find_and_extract_missing_files(self):
        """Find and extract the 4 missing original Excel files"""
        print("\n" + "="*70)
        print("ADDING MISSING ORIGINAL EXCEL FILES")
        print("="*70 + "\n")
        
        # Look for the files in data/ folder
        expected_files = [
            ('8713285.xlsx', 'xlsx'),
            ('13116880.xlsx', 'xlsx'),
            ('4505665 Trade and Cash History (2005 to 2014).xlsb', 'xlsb'),
            ('4505665 Trade History (2004).xlsb', 'xlsb')
        ]
        
        files_found = 0
        files_processed = 0
        
        for filename, file_type in expected_files:
            file_path = self.data_path / filename
            
            if not file_path.exists():
                print(f"❌ NOT FOUND: {filename}")
                print(f"   Looking in: {file_path}")
                continue
            
            files_found += 1
            print(f"\n[{files_found}/4] Found: {filename}")
            print(f"  Size: {file_path.stat().st_size / 1024:.1f} KB")
            
            # Extract based on type
            trades = []
            if file_type == 'xlsx':
                trades = self.extract_original_xlsx(file_path)
            elif file_type == 'xlsb':
                trades = self.extract_original_xlsb(file_path)
            
            if trades:
                self.new_trades.extend(trades)
                files_processed += 1
            
            print(f"  Running total: {len(self.new_trades)} new transactions")
        
        print("\n" + "="*70)
        print(f"FILES PROCESSED: {files_processed}/4 files found")
        print(f"NEW TRANSACTIONS: {len(self.new_trades)}")
        print("="*70 + "\n")
        
        return len(self.new_trades) > 0
    
    def merge_with_existing_data(self):
        """Merge new trades with existing consolidated data"""
        print("Merging with existing data...")
        
        # Read existing consolidated CSV
        existing_csv = self.data_path / 'consolidated_trades.csv'
        
        if not existing_csv.exists():
            print(f"ERROR: Existing consolidated file not found: {existing_csv}")
            return None
        
        print(f"  Reading existing data from {existing_csv.name}")
        df_existing = pd.read_csv(existing_csv, low_memory=False)

        # Convert transaction_datetime to datetime if it's a string
        if 'transaction_datetime' in df_existing.columns:
            df_existing['transaction_datetime'] = pd.to_datetime(df_existing['transaction_datetime'], errors='coerce')

        print(f"  Existing transactions: {len(df_existing)}")
        
        # Create DataFrame from new trades
        df_new = pd.DataFrame(self.new_trades)
        print(f"  New transactions: {len(df_new)}")
        
        # Parse dates for new trades
        print(f"  Parsing dates in new trades...")
        df_new['transaction_datetime'] = df_new['transaction_time'].apply(self.parse_date)
        
        # Combine
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"  Combined total: {len(df_combined)} transactions")
        
        # Re-sort by date
        df_combined_sorted = df_combined.sort_values('transaction_datetime', na_position='last')
        
        # Re-number rows
        df_combined_sorted['row_id'] = range(1, len(df_combined_sorted) + 1)
        
        return df_combined_sorted
    
    def detect_duplicates(self, df):
        """Detect potential duplicate transactions"""
        print("\nRe-running duplicate detection on full dataset...")
        
        # Create duplicate detection key
        df['dup_key'] = (
            df['transaction_datetime'].astype(str) + '_' +
            df['product'].astype(str) + '_' +
            df['type'].astype(str) + '_' +
            df['trade_value'].astype(str)
        )
        
        # Find duplicates
        duplicates = df[df.duplicated(subset=['dup_key'], keep=False)]
        
        print(f"  Found {len(duplicates)} potential duplicate rows")
        
        return duplicates
    
    def update_outputs(self, df, duplicates):
        """Update all output files with new data"""
        print("\nUpdating output files...")
        
        # 1. Update consolidated CSV
        csv_path = self.data_path / 'consolidated_trades.csv'
        df.to_csv(csv_path, index=False)
        print(f"  ✅ Updated: {csv_path}")
        
        # 2. Update duplicates report
        if len(duplicates) > 0:
            dup_path = self.data_path / 'duplicates_report.csv'
            duplicates.to_csv(dup_path, index=False)
            print(f"  ✅ Updated: {dup_path}")
        
        # 3. Update SQLite database
        db_path = self.data_path / 'trading_data.db'
        conn = sqlite3.connect(db_path)

        # Convert datetime columns to strings for SQLite compatibility
        df_for_sql = df.copy()
        if 'transaction_datetime' in df_for_sql.columns:
            df_for_sql['transaction_datetime'] = df_for_sql['transaction_datetime'].astype(str)
        if 'transaction_time' in df_for_sql.columns:
            df_for_sql['transaction_time'] = df_for_sql['transaction_time'].astype(str)

        # Replace trades table with updated data
        df_for_sql.to_sql('trades', conn, if_exists='replace', index=False)
        
        # Update duplicates table
        if len(duplicates) > 0:
            duplicates_for_sql = duplicates.copy()
            if 'transaction_datetime' in duplicates_for_sql.columns:
                duplicates_for_sql['transaction_datetime'] = duplicates_for_sql['transaction_datetime'].astype(str)
            if 'transaction_time' in duplicates_for_sql.columns:
                duplicates_for_sql['transaction_time'] = duplicates_for_sql['transaction_time'].astype(str)
            duplicates_for_sql.to_sql('potential_duplicates', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"  ✅ Updated: {db_path}")
        
        # 4. Update summary report
        self.create_updated_summary(df, duplicates)
    
    def create_updated_summary(self, df, duplicates):
        """Create updated summary report"""
        report_path = self.data_path / 'consolidation_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRADING DATA CONSOLIDATION SUMMARY (UPDATED)\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Transactions: {len(df):,}\n")
            f.write(f"  (Added {len(self.new_trades):,} from original Excel files)\n\n")
            
            # Date range
            valid_dates = df['transaction_datetime'].dropna()
            if len(valid_dates) > 0:
                f.write(f"Date Range: {valid_dates.min()} to {valid_dates.max()}\n\n")
            
            # By source type
            f.write("Transactions by Source Type:\n")
            for source_type, count in df['source_type'].value_counts().items():
                f.write(f"  {source_type}: {count:,}\n")
            f.write("\n")
            
            # Accounts
            accounts = df['account_id'].dropna().unique()
            if len(accounts) > 0:
                f.write(f"Account IDs Found: {len(accounts)}\n")
                for account in sorted([str(a) for a in accounts]):
                    account_trades = len(df[df['account_id'] == account])
                    f.write(f"  {account}: {account_trades:,} transactions\n")
                f.write("\n")
            
            # Products
            products = df['product'].value_counts().head(20)
            f.write("Top 20 Products Traded:\n")
            for product, count in products.items():
                if pd.notna(product):
                    f.write(f"  {str(product):<40} {count:>8,}\n")
            f.write("\n")
            
            # Duplicates
            f.write(f"Potential Duplicates: {len(duplicates):,}\n")
            if len(duplicates) > 0:
                f.write("  (See duplicates_report.csv for details)\n")
            f.write("\n")
            
            # Data quality
            f.write("Data Quality:\n")
            f.write(f"  Rows with valid dates: {df['transaction_datetime'].notna().sum():,} ({df['transaction_datetime'].notna().sum()/len(df)*100:.1f}%)\n")
            f.write(f"  Rows with account ID: {df['account_id'].notna().sum():,} ({df['account_id'].notna().sum()/len(df)*100:.1f}%)\n")
            f.write(f"  Rows with product: {df['product'].notna().sum():,} ({df['product'].notna().sum()/len(df)*100:.1f}%)\n")
        
        print(f"  ✅ Updated: {report_path}")


def main():
    """Main execution"""
    print("\n")
    
    data_path = Path("data")
    
    if not data_path.exists():
        print(f"ERROR: Data directory not found: {data_path}")
        sys.exit(1)
    
    # Create adder
    adder = IncrementalDataAdder(data_path)
    
    # Find and extract missing files
    success = adder.find_and_extract_missing_files()
    
    if not success:
        print("\nERROR: No files found or extracted!")
        print("\nExpected files in data/ folder:")
        print("  - 8713285.xlsx")
        print("  - 13116880.xlsx")
        print("  - 4505665 Trade and Cash History (2005 to 2014).xlsb")
        print("  - 4505665 Trade History (2004).xlsb")
        sys.exit(1)
    
    # Merge with existing data
    df_combined = adder.merge_with_existing_data()
    
    if df_combined is None:
        sys.exit(1)
    
    # Detect duplicates
    duplicates = adder.detect_duplicates(df_combined)
    
    # Update outputs
    adder.update_outputs(df_combined, duplicates)
    
    print("\n" + "="*70)
    print("UPDATE COMPLETE!")
    print("="*70)
    print(f"\nOriginal transactions: 197,103")
    print(f"Added transactions: {len(adder.new_trades):,}")
    print(f"Total transactions: {len(df_combined):,}")
    print(f"\nAll output files have been updated in: {data_path}/")
    print()


if __name__ == "__main__":
    main()