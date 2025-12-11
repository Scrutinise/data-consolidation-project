#!/usr/bin/env python3
"""
Trading Data Consolidation Script
Consolidates Excel files and converted PDF Excel files into unified database
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import io
import re

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class TradingDataConsolidator:
    """Consolidates trading data from multiple Excel sources"""
    
    def __init__(self, raw_data_path, output_path):
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.all_trades = []
        self.source_summary = []
        
    def identify_file_type(self, file_path):
        """Identify if file is original Excel or converted PDF"""
        filename = file_path.name.lower()
        
        # Original Excel files have account numbers in filename
        if any(account in filename for account in ['8713285', '13116880', '4505665']):
            if 'trade' in filename and 'history' in filename:
                return 'original_xlsb'
            else:
                return 'original_xlsx'
        
        # Converted PDFs have CMC statement naming pattern
        if filename.startswith('cmc') and file_path.suffix.lower() == '.xlsx':
            return 'converted_pdf'
        
        return 'unknown'
    
    def extract_converted_pdf_transactions(self, file_path):
        """Extract transaction data from Adobe-converted PDF Excel files"""
        print(f"  Extracting from converted PDF: {file_path.name}")
        
        try:
            # Read entire file without headers
            df_raw = pd.read_excel(file_path, sheet_name=0, header=None)
            
            # Find transaction table headers
            transaction_tables = []
            
            for idx, row in df_raw.iterrows():
                row_str = ' '.join([str(x) for x in row.dropna().values]).lower()
                
                # Look for transaction time header
                if 'transaction time' in row_str and 'type' in row_str and 'order id' in row_str:
                    print(f"    Found transaction table at row {idx}")
                    
                    # Extract column positions from header row
                    header_row = df_raw.iloc[idx]
                    
                    # Find which columns have the key headers
                    col_mapping = {}
                    for col_idx, value in enumerate(header_row):
                        if pd.notna(value):
                            val_str = str(value).strip().lower()
                            if 'transaction time' in val_str:
                                col_mapping['transaction_time'] = col_idx
                            elif val_str == 'type':
                                col_mapping['type'] = col_idx
                            elif 'order id' in val_str:
                                col_mapping['order_id'] = col_idx
                            elif 'trade id' in val_str:
                                col_mapping['trade_id'] = col_idx
                            elif 'correlated' in val_str:
                                col_mapping['correlated_order_id'] = col_idx
                            elif val_str == 'product':
                                col_mapping['product'] = col_idx
                            elif 'amt' in val_str and 'point' in val_str:
                                col_mapping['amt_per_point'] = col_idx
                            elif 'trade price' in val_str:
                                col_mapping['trade_price'] = col_idx
                            elif 'trade value' in val_str:
                                col_mapping['trade_value'] = col_idx
                            elif 'realised' in val_str and 'p&l' in val_str or 'p&l' in val_str or 'realised' in val_str:
                                col_mapping['realised_pnl'] = col_idx
                    
                    # Extract data rows after header
                    data_rows = []
                    for data_idx in range(idx + 1, len(df_raw)):
                        data_row = df_raw.iloc[data_idx]
                        
                        # Check if row has transaction data (has transaction time)
                        if 'transaction_time' in col_mapping:
                            trans_time = data_row.iloc[col_mapping['transaction_time']]
                            if pd.isna(trans_time):
                                # Empty row or end of table
                                continue
                            
                            # Check if it looks like a date/time
                            trans_time_str = str(trans_time).strip()
                            if not trans_time_str or trans_time_str == 'nan':
                                continue
                            
                            # Check if next table starts (another header row)
                            row_str_check = ' '.join([str(x) for x in data_row.dropna().values]).lower()
                            if 'transaction time' in row_str_check and 'type' in row_str_check:
                                break
                            
                            # Extract values based on column mapping
                            trade_data = {
                                'transaction_time': trans_time_str,
                                'type': str(data_row.iloc[col_mapping.get('type', 0)]) if 'type' in col_mapping else None,
                                'order_id': str(data_row.iloc[col_mapping.get('order_id', 0)]) if 'order_id' in col_mapping else None,
                                'trade_id': str(data_row.iloc[col_mapping.get('trade_id', 0)]) if 'trade_id' in col_mapping else None,
                                'correlated_order_id': str(data_row.iloc[col_mapping.get('correlated_order_id', 0)]) if 'correlated_order_id' in col_mapping else None,
                                'product': str(data_row.iloc[col_mapping.get('product', 0)]) if 'product' in col_mapping else None,
                                'amt_per_point': data_row.iloc[col_mapping.get('amt_per_point', 0)] if 'amt_per_point' in col_mapping else None,
                                'trade_price': data_row.iloc[col_mapping.get('trade_price', 0)] if 'trade_price' in col_mapping else None,
                                'trade_value': data_row.iloc[col_mapping.get('trade_value', 0)] if 'trade_value' in col_mapping else None,
                                'realised_pnl': data_row.iloc[col_mapping.get('realised_pnl', 0)] if 'realised_pnl' in col_mapping else None,
                            }
                            
                            # Clean up 'nan' strings
                            for key, value in trade_data.items():
                                if value == 'nan' or (isinstance(value, float) and pd.isna(value)):
                                    trade_data[key] = None
                            
                            data_rows.append(trade_data)
                    
                    if data_rows:
                        print(f"    Extracted {len(data_rows)} transactions from this table")
                        transaction_tables.extend(data_rows)
            
            if transaction_tables:
                print(f"  Total transactions extracted: {len(transaction_tables)}")
                return transaction_tables
            else:
                print(f"  WARNING: No transaction tables found in {file_path.name}")
                return []
                
        except Exception as e:
            print(f"  ERROR extracting from {file_path.name}: {str(e)}")
            return []
    
    def extract_original_xlsx(self, file_path):
        """Extract data from original XLSX files (Schema A format)"""
        print(f"  Reading original XLSX: {file_path.name}")
        
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            
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
                }
                trades.append(trade)
            
            print(f"  Extracted {len(trades)} transactions")
            return trades
            
        except Exception as e:
            print(f"  ERROR reading {file_path.name}: {str(e)}")
            return []
    
    def extract_original_xlsb(self, file_path):
        """Extract data from original XLSB files (Schema B format)"""
        print(f"  Reading original XLSB: {file_path.name}")
        
        try:
            df = pd.read_excel(file_path, engine='pyxlsb')
            
            trades = []
            
            # Determine which schema based on columns
            if 'Trade Time' in df.columns:
                # Schema B1: Trade History format
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
                    }
                    trades.append(trade)
            
            elif 'Time' in df.columns and 'Transaction Type' in df.columns:
                # Schema B2: Trade and Cash History format
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
                    }
                    trades.append(trade)
            
            print(f"  Extracted {len(trades)} transactions")
            return trades
            
        except Exception as e:
            print(f"  ERROR reading {file_path.name}: {str(e)}")
            return []
    
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
                # Might be microseconds or other format
                try:
                    return pd.Timestamp(date_value, unit='us')
                except:
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
    
    def consolidate_all_files(self):
        """Main consolidation process"""
        print("\n" + "="*70)
        print("TRADING DATA CONSOLIDATION")
        print("="*70 + "\n")
        
        # Get all Excel files
        xlsx_files = sorted(self.raw_data_path.glob('*.xlsx'))
        xlsb_files = sorted(self.raw_data_path.glob('*.xlsb'))
        
        all_files = xlsx_files + xlsb_files
        
        print(f"Found {len(xlsx_files)} XLSX files and {len(xlsb_files)} XLSB files")
        print(f"Total files to process: {len(all_files)}\n")
        print("="*70 + "\n")
        
        # Process each file
        for idx, file_path in enumerate(all_files, 1):
            print(f"[{idx}/{len(all_files)}] Processing: {file_path.name}")
            
            file_type = self.identify_file_type(file_path)
            print(f"  File type: {file_type}")
            
            trades = []
            
            if file_type == 'converted_pdf':
                trades = self.extract_converted_pdf_transactions(file_path)
            elif file_type == 'original_xlsx':
                trades = self.extract_original_xlsx(file_path)
            elif file_type == 'original_xlsb':
                trades = self.extract_original_xlsb(file_path)
            else:
                print(f"  WARNING: Unknown file type, skipping")
                continue
            
            # Add source information to each trade
            for trade in trades:
                trade['source_file'] = file_path.name
                trade['source_type'] = file_type
            
            self.all_trades.extend(trades)
            
            # Track summary
            self.source_summary.append({
                'filename': file_path.name,
                'file_type': file_type,
                'transactions': len(trades)
            })
            
            print(f"  Running total: {len(self.all_trades)} transactions\n")
        
        print("="*70)
        print(f"EXTRACTION COMPLETE: {len(self.all_trades)} total transactions")
        print("="*70 + "\n")
    
    def create_unified_dataframe(self):
        """Create unified pandas DataFrame from all trades"""
        print("Creating unified DataFrame...")
        
        if not self.all_trades:
            print("ERROR: No trades to consolidate!")
            return None
        
        df = pd.DataFrame(self.all_trades)
        
        # Parse dates
        print("  Parsing dates...")
        df['transaction_datetime'] = df['transaction_time'].apply(self.parse_date)
        
        # Sort by date
        df_sorted = df.sort_values('transaction_datetime', na_position='last')
        
        # Add row number
        df_sorted['row_id'] = range(1, len(df_sorted) + 1)
        
        print(f"  Unified DataFrame created: {len(df_sorted)} rows, {len(df_sorted.columns)} columns")
        
        return df_sorted
    
    def detect_duplicates(self, df):
        """Detect potential duplicate transactions"""
        print("\nDetecting duplicates...")
        
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
    
    def create_outputs(self, df, duplicates):
        """Create all output files"""
        print("\nCreating output files...")
        
        # 1. Consolidated CSV
        csv_path = self.output_path / 'consolidated_trades.csv'
        df.to_csv(csv_path, index=False)
        print(f"  Created: {csv_path}")
        
        # 2. Duplicates report
        if len(duplicates) > 0:
            dup_path = self.output_path / 'duplicates_report.csv'
            duplicates.to_csv(dup_path, index=False)
            print(f"  Created: {dup_path}")
        
        # 3. SQLite database
        db_path = self.output_path / 'trading_data.db'
        conn = sqlite3.connect(db_path)

        # Convert datetime columns to strings for SQLite compatibility
        df_for_sql = df.copy()
        if 'transaction_datetime' in df_for_sql.columns:
            df_for_sql['transaction_datetime'] = df_for_sql['transaction_datetime'].astype(str)
        if 'transaction_time' in df_for_sql.columns:
            df_for_sql['transaction_time'] = df_for_sql['transaction_time'].astype(str)

        # Main trades table
        df_for_sql.to_sql('trades', conn, if_exists='replace', index=False)

        # Source summary table
        pd.DataFrame(self.source_summary).to_sql('source_files', conn, if_exists='replace', index=False)

        # Duplicates table
        if len(duplicates) > 0:
            duplicates_for_sql = duplicates.copy()
            if 'transaction_datetime' in duplicates_for_sql.columns:
                duplicates_for_sql['transaction_datetime'] = duplicates_for_sql['transaction_datetime'].astype(str)
            if 'transaction_time' in duplicates_for_sql.columns:
                duplicates_for_sql['transaction_time'] = duplicates_for_sql['transaction_time'].astype(str)
            duplicates_for_sql.to_sql('potential_duplicates', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"  Created: {db_path}")
        
        # 4. Summary report
        self.create_summary_report(df, duplicates)
    
    def create_summary_report(self, df, duplicates):
        """Create text summary report"""
        report_path = self.output_path / 'consolidation_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRADING DATA CONSOLIDATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Transactions: {len(df):,}\n\n")
            
            # Date range
            valid_dates = df['transaction_datetime'].dropna()
            if len(valid_dates) > 0:
                f.write(f"Date Range: {valid_dates.min()} to {valid_dates.max()}\n\n")
            
            # By source type
            f.write("Transactions by Source Type:\n")
            for source_type, count in df['source_type'].value_counts().items():
                f.write(f"  {source_type}: {count:,}\n")
            f.write("\n")
            
            # By file
            f.write("Transactions by File:\n")
            for _, row in pd.DataFrame(self.source_summary).sort_values('transactions', ascending=False).iterrows():
                f.write(f"  {row['filename']:<50} {row['transactions']:>8,}\n")
            f.write("\n")
            
            # Accounts
            if 'account_id' in df.columns:
                accounts = df['account_id'].dropna().unique()
                if len(accounts) > 0:
                    f.write(f"Account IDs: {', '.join([str(a) for a in accounts])}\n\n")
            
            # Products
            products = df['product'].value_counts().head(20)
            f.write("Top 20 Products Traded:\n")
            for product, count in products.items():
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
            if 'account_id' in df.columns:
                f.write(f"  Rows with account ID: {df['account_id'].notna().sum():,} ({df['account_id'].notna().sum()/len(df)*100:.1f}%)\n")
            f.write(f"  Rows with product: {df['product'].notna().sum():,} ({df['product'].notna().sum()/len(df)*100:.1f}%)\n")
        
        print(f"  Created: {report_path}")


def main():
    """Main execution"""
    print("\n")
    
    # Paths
    raw_data_path = Path("data/raw")
    output_path = Path("data")
    
    # Check raw data exists
    if not raw_data_path.exists():
        print(f"ERROR: Directory not found: {raw_data_path}")
        print(f"Current working directory: {Path.cwd()}")
        sys.exit(1)
    
    # Create consolidator
    consolidator = TradingDataConsolidator(raw_data_path, output_path)
    
    # Run consolidation
    consolidator.consolidate_all_files()
    
    # Create unified DataFrame
    df = consolidator.create_unified_dataframe()
    
    if df is None:
        sys.exit(1)
    
    # Detect duplicates
    duplicates = consolidator.detect_duplicates(df)
    
    # Create outputs
    consolidator.create_outputs(df, duplicates)
    
    print("\n" + "="*70)
    print("CONSOLIDATION COMPLETE!")
    print("="*70)
    print(f"\nOutput files created in: {output_path}/")
    print(f"  - consolidated_trades.csv")
    print(f"  - trading_data.db")
    print(f"  - consolidation_summary.txt")
    if len(duplicates) > 0:
        print(f"  - duplicates_report.csv")
    print()


if __name__ == "__main__":
    main()