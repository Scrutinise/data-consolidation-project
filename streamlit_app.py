"""
Trading Performance & Behavioral Analysis Dashboard
Streamlit app for analyzing trading patterns and financial health
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import numpy as np
from anthropic import Anthropic
import streamlit as st
import hashlib

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == st.secrets.get("password_hash", ""):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # Return True if password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

# Add this right after the function definition
if not check_password():
    st.stop()  # Don't run the rest of the app

# Page config
st.set_page_config(
    page_title="Trading Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .alert-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_db_connection():
    """Initialize database connection"""
    return sqlite3.connect('data/trading_data.db', check_same_thread=False)


@st.cache_data(ttl=600)
def load_data(_conn):
    """Load trading data from database"""
    query = "SELECT * FROM trades"
    df = pd.read_sql_query(query, _conn)
    
    # Parse dates
    df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'], errors='coerce')
    
    # Convert numeric columns
    numeric_cols = ['amt_per_point', 'trade_price', 'trade_value', 'realised_pnl', 
                    'amount_ccy', 'cf_balance', 'quantity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def calculate_cash_balance_timeline(df):
    """Calculate account cash balance over time"""
    # Use C/F BALANCE if available (from original Excel files)
    if 'cf_balance' in df.columns:
        balance_df = df[['transaction_datetime', 'cf_balance', 'account_id']].dropna(subset=['cf_balance'])
        balance_df = balance_df.sort_values('transaction_datetime')
        return balance_df
    
    # Otherwise calculate from transactions
    df_sorted = df.sort_values('transaction_datetime').copy()
    
    # Estimate balance changes
    df_sorted['balance_change'] = 0
    
    # Realized P&L affects balance
    if 'realised_pnl' in df.columns:
        df_sorted['balance_change'] += df_sorted['realised_pnl'].fillna(0)
    
    # AMOUNT affects balance (deposits/withdrawals)
    if 'trade_value' in df.columns:
        # Negative values might be withdrawals
        df_sorted.loc[df_sorted['type'].str.contains('withdrawal', case=False, na=False), 'balance_change'] -= df_sorted['trade_value'].abs()
        df_sorted.loc[df_sorted['type'].str.contains('deposit|payment', case=False, na=False), 'balance_change'] += df_sorted['trade_value'].abs()
    
    # Calculate cumulative balance by account
    df_sorted['estimated_balance'] = df_sorted.groupby('account_id')['balance_change'].cumsum()
    
    return df_sorted[['transaction_datetime', 'estimated_balance', 'account_id']]


def identify_deposits_withdrawals(df):
    """Identify deposits and withdrawals from transaction data"""
    deposits_withdrawals = []
    
    # Check booking type column
    if 'type' in df.columns:
        deposit_keywords = ['payment', 'deposit', 'transfer in', 'credit']
        withdrawal_keywords = ['withdrawal', 'withdraw', 'transfer out', 'debit']
        
        for keyword in deposit_keywords:
            mask = df['type'].str.contains(keyword, case=False, na=False)
            deposits_withdrawals.append(df[mask].copy())
        
        for keyword in withdrawal_keywords:
            mask = df['type'].str.contains(keyword, case=False, na=False)
            temp = df[mask].copy()
            temp['trade_value'] = -temp['trade_value'].abs()  # Make withdrawals negative
            deposits_withdrawals.append(temp)
    
    if deposits_withdrawals:
        result = pd.concat(deposits_withdrawals, ignore_index=True)
        result = result.sort_values('transaction_datetime')
        return result[['transaction_datetime', 'type', 'trade_value', 'account_id']]
    
    return pd.DataFrame()


def calculate_stake_size_over_time(df):
    """Calculate average stake size over time (rolling window)"""
    # Use amt_per_point or trade_value as stake size indicator
    stake_col = 'amt_per_point' if 'amt_per_point' in df.columns else 'trade_value'
    
    df_stakes = df[['transaction_datetime', stake_col, 'account_id']].dropna()
    df_stakes = df_stakes.sort_values('transaction_datetime')
    
    # Calculate rolling average stake size (30-day window)
    df_stakes['rolling_avg_stake'] = df_stakes.groupby('account_id')[stake_col].transform(
        lambda x: x.rolling(window=50, min_periods=1).mean()
    )
    
    return df_stakes


def detect_loss_chasing(df):
    """Detect patterns of increased betting after losses"""
    df_sorted = df.sort_values('transaction_datetime').copy()
    
    # Focus on trades with P&L
    if 'realised_pnl' not in df.columns:
        return pd.DataFrame()
    
    df_pnl = df_sorted[df_sorted['realised_pnl'].notna()].copy()
    
    # Identify losses
    df_pnl['is_loss'] = df_pnl['realised_pnl'] < 0
    
    # Get stake size
    stake_col = 'amt_per_point' if 'amt_per_point' in df.columns else 'trade_value'
    
    # Calculate next trade stake after loss
    df_pnl['prev_was_loss'] = df_pnl.groupby('account_id')['is_loss'].shift(1)
    df_pnl['prev_stake'] = df_pnl.groupby('account_id')[stake_col].shift(1)
    df_pnl['stake_increase'] = ((df_pnl[stake_col] - df_pnl['prev_stake']) / df_pnl['prev_stake'] * 100).fillna(0)
    
    # Flag cases where stake increased significantly after a loss
    chasing_threshold = 20  # 20% stake increase
    df_pnl['possible_chasing'] = (df_pnl['prev_was_loss'] == True) & (df_pnl['stake_increase'] > chasing_threshold)
    
    return df_pnl[['transaction_datetime', 'prev_was_loss', 'stake_increase', 'possible_chasing', 'account_id']]


def calculate_session_patterns(df):
    """Analyze trading session patterns"""
    df_sorted = df.sort_values('transaction_datetime').copy()
    
    # Calculate time between trades
    df_sorted['time_since_last_trade'] = df_sorted.groupby('account_id')['transaction_datetime'].diff()
    
    # Define session break (more than 4 hours = new session)
    session_break = pd.Timedelta(hours=4)
    df_sorted['new_session'] = df_sorted['time_since_last_trade'] > session_break
    df_sorted['session_id'] = df_sorted.groupby('account_id')['new_session'].cumsum()
    
    # Calculate session statistics
    session_stats = df_sorted.groupby(['account_id', 'session_id']).agg({
        'transaction_datetime': ['min', 'max', 'count'],
        'realised_pnl': 'sum'
    }).reset_index()
    
    session_stats.columns = ['account_id', 'session_id', 'session_start', 'session_end', 'trades_in_session', 'session_pnl']
    session_stats['session_duration'] = session_stats['session_end'] - session_stats['session_start']
    session_stats['session_duration_hours'] = session_stats['session_duration'].dt.total_seconds() / 3600
    
    return session_stats


def create_balance_chart(balance_df):
    """Create account balance over time chart"""
    fig = go.Figure()
    
    if 'cf_balance' in balance_df.columns:
        balance_col = 'cf_balance'
        title = "Account Cash Balance Over Time"
    else:
        balance_col = 'estimated_balance'
        title = "Estimated Account Balance Over Time"
    
    # Plot by account
    for account in balance_df['account_id'].dropna().unique():
        account_data = balance_df[balance_df['account_id'] == account]
        fig.add_trace(go.Scatter(
            x=account_data['transaction_datetime'],
            y=account_data[balance_col],
            mode='lines',
            name=f'Account {account}',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Balance (£)",
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_deposits_withdrawals_chart(dw_df):
    """Create deposits vs withdrawals chart - withdrawals shown as negative bars"""
    if dw_df.empty:
        return None
    
    fig = go.Figure()
    
    # Separate deposits and withdrawals
    deposits = dw_df[dw_df['trade_value'] > 0]
    withdrawals = dw_df[dw_df['trade_value'] < 0]
    
    # Add deposits (green bars going UP)
    fig.add_trace(go.Bar(
        x=deposits['transaction_datetime'],
        y=deposits['trade_value'],
        name='Deposits',
        marker_color='green',
        opacity=0.7,
        hovertemplate='<b>Deposit</b><br>Date: %{x}<br>Amount: £%{y:,.2f}<extra></extra>'
    ))
    
    # Add withdrawals (red bars going DOWN - negative values)
    fig.add_trace(go.Bar(
        x=withdrawals['transaction_datetime'],
        y=withdrawals['trade_value'],  # Keep negative - bars go down
        name='Withdrawals',
        marker_color='red',
        opacity=0.7,
        hovertemplate='<b>Withdrawal</b><br>Date: %{x}<br>Amount: £%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Deposits and Withdrawals Over Time",
        xaxis_title="Date",
        yaxis_title="Amount (£)",
        barmode='overlay',
        hovermode='closest',
        height=500,
        yaxis=dict(
            range=[-5000, 5000],  # FIXED: -£5,000 to +£5,000
            zeroline=True,
            zerolinewidth=3,
            zerolinecolor='black',
            gridcolor='lightgray',
            tickformat=',.0f',
            tickprefix='£'
        ),
        showlegend=True
    )
    
    return fig

def create_stake_size_chart(stake_df):
    """Create stake size progression chart"""
    fig = go.Figure()
    
    for account in stake_df['account_id'].dropna().unique():
        account_data = stake_df[stake_df['account_id'] == account]
        fig.add_trace(go.Scatter(
            x=account_data['transaction_datetime'],
            y=account_data['rolling_avg_stake'],
            mode='lines',
            name=f'Account {account}',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Average Stake Size Over Time (50-trade rolling average)",
        xaxis_title="Date",
        yaxis_title="Average Stake Size",
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_loss_chasing_chart(chasing_df):
    """Create loss chasing detection chart"""
    if chasing_df.empty:
        return None
    
    chasing_events = chasing_df[chasing_df['possible_chasing'] == True]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=chasing_events['transaction_datetime'],
        y=chasing_events['stake_increase'],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='triangle-up'
        ),
        name='Possible Loss Chasing',
        text=chasing_events['stake_increase'].apply(lambda x: f'+{x:.1f}%'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Stake Increase:</b> %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Detected Loss-Chasing Events (Stake increased >20% after loss)",
        xaxis_title="Date",
        yaxis_title="Stake Increase (%)",
        hovermode='closest',
        height=400
    )
    
    return fig


def create_activity_heatmap(df):
    """Create hour-of-day / day-of-week activity heatmap"""
    df_time = df.copy()
    df_time['hour'] = df_time['transaction_datetime'].dt.hour
    df_time['day_of_week'] = df_time['transaction_datetime'].dt.day_name()
    
    # Count trades by hour and day
    heatmap_data = df_time.groupby(['day_of_week', 'hour']).size().reset_index(name='trade_count')
    
    # Pivot for heatmap
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='trade_count').fillna(0)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex([day for day in day_order if day in heatmap_pivot.index])
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='Reds',
        hovertemplate='<b>%{y}</b><br>Hour: %{x}:00<br>Trades: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Trading Activity Heatmap (Hour of Day vs Day of Week)",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=400
    )
    
    return fig


def query_with_claude(question, df, api_key):
    """Query the data using Claude API"""
    try:
        client = Anthropic(api_key=api_key)
        
        # Get data summary for context
        data_summary = f"""
Database contains {len(df)} trading transactions.
Columns: {', '.join(df.columns)}
Date range: {df['transaction_datetime'].min()} to {df['transaction_datetime'].max()}
Accounts: {', '.join(df['account_id'].dropna().unique().astype(str))}

Sample data:
{df.head(5).to_string()}
        """
        
        prompt = f"""You are a trading data analyst. You have access to a SQLite database with trading data.

{data_summary}

User question: {question}

Please provide:
1. A SQL query to answer this question
2. An explanation of what the query does
3. Insights based on the data

Format your response clearly with:
- SQL Query: (the query)
- Explanation: (what it does)
- Insights: (analysis)
"""
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
        
    except Exception as e:
        return f"Error querying Claude API: {str(e)}"


# Main App
def main():
    st.markdown('<p class="main-header">📊 Trading Performance & Behavioral Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "📈 Overview",
        "💰 Financial Health",
        "🎯 Behavioral Patterns",
        "🚨 Risk Indicators",
        "📊 Product Analysis",
        "💬 Ask Questions"
    ])
    
    # Load data
    conn = init_db_connection()
    df = load_data(conn)
    
    # Filters in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    
    # Account filter
    accounts = ['All'] + sorted(df['account_id'].dropna().unique().astype(str).tolist())
    selected_account = st.sidebar.selectbox("Account", accounts)
    
    # Date filter
    min_date = df['transaction_datetime'].min().date()
    max_date = df['transaction_datetime'].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Apply filters
    df_filtered = df.copy()
    if selected_account != 'All':
        df_filtered = df_filtered[df_filtered['account_id'] == selected_account]
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['transaction_datetime'].dt.date >= date_range[0]) &
            (df_filtered['transaction_datetime'].dt.date <= date_range[1])
        ]
    
    # Page content
    if page == "📈 Overview":
        show_overview_page(df_filtered)
    elif page == "💰 Financial Health":
        show_financial_health_page(df_filtered)
    elif page == "🎯 Behavioral Patterns":
        show_behavioral_patterns_page(df_filtered)
    elif page == "🚨 Risk Indicators":
        show_risk_indicators_page(df_filtered)
    elif page == "📊 Product Analysis":
        show_product_analysis_page(df_filtered)
    elif page == "💬 Ask Questions":
        show_questions_page(df_filtered, conn)


def show_overview_page(df):
    """Display overview dashboard"""
    st.header("Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", f"{len(df):,}")
    
    with col2:
        if df['transaction_datetime'].notna().any():
            date_range_str = f"{df['transaction_datetime'].min().strftime('%Y')} - {df['transaction_datetime'].max().strftime('%Y')}"
            st.metric("Date Range", date_range_str)
    
    with col3:
        num_accounts = df['account_id'].nunique()
        st.metric("Accounts", num_accounts)
    
    with col4:
        num_products = df['product'].nunique()
        st.metric("Products Traded", num_products)
    
    st.markdown("---")
    
    # Trading volume over time
    st.subheader("Trading Volume Over Time")
    volume_data = df.groupby(df['transaction_datetime'].dt.to_period('M')).size().reset_index(name='trades')
    volume_data['transaction_datetime'] = volume_data['transaction_datetime'].dt.to_timestamp()
    
    fig = px.line(volume_data, x='transaction_datetime', y='trades', 
                  title='Monthly Trading Volume',
                  labels={'transaction_datetime': 'Month', 'trades': 'Number of Trades'})
    fig.update_traces(line_color='#1f77b4', line_width=3)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top products
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Products by Volume")
        top_products = df['product'].value_counts().head(10)
        fig = px.bar(x=top_products.values, y=top_products.index, orientation='h',
                     labels={'x': 'Number of Trades', 'y': 'Product'})
        fig.update_traces(marker_color='#1f77b4')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Account Distribution")
        account_dist = df['account_id'].value_counts()
        fig = px.pie(values=account_dist.values, names=account_dist.index,
                     title='Trades by Account')
        st.plotly_chart(fig, use_container_width=True)


def show_financial_health_page(df):
    """Display financial health analysis"""
    st.header("💰 Financial Health Analysis")
    
    # Calculate balance timeline
    balance_df = calculate_cash_balance_timeline(df)
    
    if not balance_df.empty:
        st.subheader("Account Balance Over Time")
        balance_chart = create_balance_chart(balance_df)
        st.plotly_chart(balance_chart, use_container_width=True)
        
        # Balance statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'cf_balance' in balance_df.columns:
                current_balance = balance_df['cf_balance'].iloc[-1]
                starting_balance = balance_df['cf_balance'].iloc[0]
            else:
                current_balance = balance_df['estimated_balance'].iloc[-1]
                starting_balance = balance_df['estimated_balance'].iloc[0]
            
            st.metric("Current Balance", f"£{current_balance:,.2f}")
        
        with col2:
            net_change = current_balance - starting_balance
            st.metric("Net Change", f"£{net_change:,.2f}", 
                     delta=f"{(net_change/starting_balance*100):.1f}%" if starting_balance != 0 else "N/A")
        
        with col3:
            if 'cf_balance' in balance_df.columns:
                max_balance = balance_df['cf_balance'].max()
            else:
                max_balance = balance_df['estimated_balance'].max()
            st.metric("Peak Balance", f"£{max_balance:,.2f}")
    
    st.markdown("---")
    
    # Deposits and Withdrawals
    st.subheader("Deposits and Withdrawals")
    dw_df = identify_deposits_withdrawals(df)
    
    if not dw_df.empty:
        dw_chart = create_deposits_withdrawals_chart(dw_df)
        if dw_chart:
            st.plotly_chart(dw_chart, use_container_width=True)
        
        # D&W statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_deposits = dw_df[dw_df['trade_value'] > 0]['trade_value'].sum()
            st.metric("Total Deposits", f"£{total_deposits:,.2f}")
        
        with col2:
            total_withdrawals = dw_df[dw_df['trade_value'] < 0]['trade_value'].abs().sum()
            st.metric("Total Withdrawals", f"£{total_withdrawals:,.2f}")
        
        with col3:
            net_cash_flow = total_deposits - total_withdrawals
            st.metric("Net Cash Flow", f"£{net_cash_flow:,.2f}",
                     delta="Positive" if net_cash_flow > 0 else "Negative")
    else:
        st.info("No clear deposits/withdrawals detected in data")


def show_behavioral_patterns_page(df):
    """Display behavioral pattern analysis"""
    st.header("🎯 Behavioral Pattern Analysis")
    
    # Stake size over time
    st.subheader("Stake Size Progression")
    stake_df = calculate_stake_size_over_time(df)
    
    if not stake_df.empty:
        stake_chart = create_stake_size_chart(stake_df)
        st.plotly_chart(stake_chart, use_container_width=True)
        
        # Check for escalation
        recent_avg = stake_df['rolling_avg_stake'].tail(100).mean()
        early_avg = stake_df['rolling_avg_stake'].head(100).mean()
        
        if recent_avg > early_avg * 1.5:
            st.markdown('<div class="alert-card">⚠️ <b>Alert:</b> Stake sizes have increased by more than 50% over time</div>', 
                       unsafe_allow_html=True)
        elif recent_avg > early_avg * 1.2:
            st.markdown('<div class="warning-card">⚠️ <b>Notice:</b> Stake sizes have increased by more than 20% over time</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-card">✅ Stake sizes remain relatively stable</div>', 
                       unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Loss chasing detection
    st.subheader("Loss Recovery Patterns")
    chasing_df = detect_loss_chasing(df)
    
    if not chasing_df.empty and chasing_df['possible_chasing'].any():
        chasing_chart = create_loss_chasing_chart(chasing_df)
        if chasing_chart:
            st.plotly_chart(chasing_chart, use_container_width=True)
        
        num_chasing_events = chasing_df['possible_chasing'].sum()
        chasing_rate = (num_chasing_events / len(chasing_df)) * 100
        
        if chasing_rate > 10:
            st.markdown(f'<div class="alert-card">🚨 <b>High Risk:</b> {num_chasing_events} potential loss-chasing events detected ({chasing_rate:.1f}% of trades)</div>', 
                       unsafe_allow_html=True)
        elif chasing_rate > 5:
            st.markdown(f'<div class="warning-card">⚠️ <b>Moderate Risk:</b> {num_chasing_events} potential loss-chasing events detected ({chasing_rate:.1f}% of trades)</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-card">✅ Low chasing behavior detected ({num_chasing_events} events, {chasing_rate:.1f}%)</div>', 
                       unsafe_allow_html=True)
    else:
        st.info("Insufficient P&L data to analyze loss-chasing patterns")
    
    st.markdown("---")
    
    # Activity heatmap
    st.subheader("Trading Activity Patterns")
    heatmap = create_activity_heatmap(df)
    st.plotly_chart(heatmap, use_container_width=True)
    
    # Late night/weekend trading check
    df_time = df.copy()
    df_time['hour'] = df_time['transaction_datetime'].dt.hour
    df_time['is_weekend'] = df_time['transaction_datetime'].dt.dayofweek >= 5
    df_time['is_late_night'] = (df_time['hour'] >= 22) | (df_time['hour'] <= 5)
    
    weekend_trades = df_time['is_weekend'].sum()
    late_night_trades = df_time['is_late_night'].sum()
    
    col1, col2 = st.columns(2)
    with col1:
        weekend_pct = (weekend_trades / len(df_time)) * 100
        st.metric("Weekend Trading", f"{weekend_pct:.1f}%", 
                 delta=f"{weekend_trades:,} trades")
    
    with col2:
        late_night_pct = (late_night_trades / len(df_time)) * 100
        st.metric("Late Night Trading (10pm-5am)", f"{late_night_pct:.1f}%",
                 delta=f"{late_night_trades:,} trades")
    
    if weekend_pct > 30 or late_night_pct > 20:
        st.markdown('<div class="warning-card">⚠️ <b>Notice:</b> Significant trading activity during high-risk times (weekends/late nights)</div>', 
                   unsafe_allow_html=True)


def show_risk_indicators_page(df):
    """Display risk indicator dashboard"""
    st.header("🚨 Risk Indicator Dashboard")
    
    # Session analysis
    st.subheader("Trading Session Analysis")
    session_stats = calculate_session_patterns(df)
    
    if not session_stats.empty:
        # Long sessions
        long_sessions = session_stats[session_stats['session_duration_hours'] > 3]
        avg_session_duration = session_stats['session_duration_hours'].mean()
        max_session_duration = session_stats['session_duration_hours'].max()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Sessions", len(session_stats))
        
        with col2:
            st.metric("Avg Session Duration", f"{avg_session_duration:.1f} hours")
        
        with col3:
            st.metric("Longest Session", f"{max_session_duration:.1f} hours")
        
        if len(long_sessions) > 0:
            st.markdown(f'<div class="warning-card">⚠️ {len(long_sessions)} sessions exceeded 3 hours</div>', 
                       unsafe_allow_html=True)
        
        # Session duration distribution
        fig = px.histogram(session_stats, x='session_duration_hours', 
                          nbins=30,
                          title='Session Duration Distribution',
                          labels={'session_duration_hours': 'Session Duration (hours)'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Consecutive loss streaks
    st.subheader("Loss Streak Analysis")
    
    if 'realised_pnl' in df.columns:
        df_pnl = df[df['realised_pnl'].notna()].sort_values('transaction_datetime').copy()
        df_pnl['is_loss'] = df_pnl['realised_pnl'] < 0
        
        # Calculate streaks
        df_pnl['streak_id'] = (df_pnl['is_loss'] != df_pnl['is_loss'].shift()).cumsum()
        loss_streaks = df_pnl[df_pnl['is_loss']].groupby('streak_id').size()
        
        if len(loss_streaks) > 0:
            max_loss_streak = loss_streaks.max()
            avg_loss_streak = loss_streaks.mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Longest Loss Streak", f"{max_loss_streak} trades")
            
            with col2:
                st.metric("Average Loss Streak", f"{avg_loss_streak:.1f} trades")
            
            if max_loss_streak > 10:
                st.markdown(f'<div class="alert-card">🚨 <b>Alert:</b> Experienced a loss streak of {max_loss_streak} consecutive trades</div>', 
                           unsafe_allow_html=True)
            
            # Streak distribution
            fig = px.histogram(loss_streaks, 
                              title='Loss Streak Length Distribution',
                              labels={'value': 'Streak Length', 'count': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Break periods
    st.subheader("Trading Break Analysis")
    df_sorted = df.sort_values('transaction_datetime').copy()
    df_sorted['days_since_last_trade'] = df_sorted.groupby('account_id')['transaction_datetime'].diff().dt.total_seconds() / 86400
    
    breaks_df = df_sorted[df_sorted['days_since_last_trade'] > 7]  # Breaks longer than a week
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Week+ Breaks", len(breaks_df))
    
    with col2:
        if not breaks_df.empty:
            avg_break = breaks_df['days_since_last_trade'].mean()
            st.metric("Average Break Length", f"{avg_break:.1f} days")
    
    if len(breaks_df) < 10 and len(df) > 1000:
        st.markdown('<div class="warning-card">⚠️ <b>Notice:</b> Few extended breaks detected - consider regular trading pauses</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-card">✅ Regular breaks observed in trading pattern</div>', 
                   unsafe_allow_html=True)


def show_product_analysis_page(df):
    """Display product performance analysis"""
    st.header("📊 Product Analysis")
    
    # Product selection
    products = df['product'].dropna().unique()
    selected_product = st.selectbox("Select Product", ['All'] + sorted(products.tolist()))
    
    if selected_product != 'All':
        df_product = df[df['product'] == selected_product]
    else:
        df_product = df
    
    # Product metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Trades", len(df_product))
    
    with col2:
        if 'realised_pnl' in df_product.columns:
            total_pnl = df_product['realised_pnl'].sum()
            st.metric("Total P&L", f"£{total_pnl:,.2f}")
    
    with col3:
        if 'realised_pnl' in df_product.columns:
            wins = (df_product['realised_pnl'] > 0).sum()
            total_with_pnl = df_product['realised_pnl'].notna().sum()
            win_rate = (wins / total_with_pnl * 100) if total_with_pnl > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
    
    st.markdown("---")
    
    # Product comparison
    st.subheader("Product Performance Comparison")
    
    product_stats = df.groupby('product').agg({
        'product': 'count',
        'realised_pnl': 'sum'
    }).rename(columns={'product': 'trade_count'}).reset_index()
    
    product_stats = product_stats.nlargest(15, 'trade_count')
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Trade Volume', 'Total P&L'),
                        specs=[[{'type': 'bar'}, {'type': 'bar'}]])
    
    fig.add_trace(
        go.Bar(x=product_stats['product'], y=product_stats['trade_count'], name='Trades'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=product_stats['product'], y=product_stats['realised_pnl'], 
               name='P&L',
               marker_color=product_stats['realised_pnl'].apply(lambda x: 'green' if x > 0 else 'red')),
        row=1, col=2
    )
    
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(height=500, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)


def show_questions_page(df, conn):
    """Display enhanced query page with SQL executor and visual query builder"""
    st.header("💬 Ask Questions About Your Data")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'last_sql' not in st.session_state:
        st.session_state.last_sql = ""
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None

    # Create tabs for different query methods
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Natural Language",
        "🔍 SQL Executor",
        "📊 Visual Query Builder",
        "📚 Query Library"
    ])

    # =========================================================================
    # TAB 1: NATURAL LANGUAGE QUESTIONS (Enhanced)
    # =========================================================================
    with tab1:
        st.markdown("""
        Ask questions in plain English and Claude will analyze your trading data.

        **Examples:**
        - "What was my most profitable month?"
        - "Show me all GBP/JPY trades where I lost more than £500"
        - "Which products have the best win rate?"
        - "Find patterns in my weekend trading"
        """)

        # API Key check
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")

        if not api_key:
            st.info("💡 Admin: Add ANTHROPIC_API_KEY to Streamlit Secrets for automatic queries")
            api_key = st.text_input("Or enter Claude API Key manually",
                                   type="password",
                                   help="Get your API key from console.anthropic.com")
        else:
            st.success("✅ Claude API connected")

        # Question input
        question = st.text_area("Your Question:", height=100, key="nl_question")

        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("Ask Claude", type="primary")
        with col2:
            if st.button("Clear History"):
                st.session_state.conversation_history = []
                st.rerun()

        if ask_button:
            if not api_key:
                st.warning("Please enter your Claude API key")
            elif not question:
                st.warning("Please enter a question")
            else:
                with st.spinner("Analyzing your data..."):
                    response = query_with_claude(question, df, api_key)

                    # Save to history
                    st.session_state.conversation_history.append({
                        'question': question,
                        'response': response,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

                    st.markdown("### 💬 Response:")
                    st.markdown(response)

                    # Extract SQL if present in response
                    import re
                    sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL)
                    if sql_match:
                        extracted_sql = sql_match.group(1).strip()
                        st.session_state.last_sql = extracted_sql

                        st.markdown("---")
                        st.markdown("**SQL Query Found in Response:**")
                        st.code(extracted_sql, language="sql")

                        if st.button("🚀 Run This Query", key="run_extracted_sql"):
                            try:
                                result_df = pd.read_sql_query(extracted_sql, conn)
                                st.session_state.last_results = result_df
                                st.success(f"✅ Query executed! {len(result_df)} rows returned")
                                st.dataframe(result_df, use_container_width=True)

                                # Download button
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    "💾 Download Results",
                                    csv,
                                    f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv"
                                )
                            except Exception as e:
                                st.error(f"Error executing query: {str(e)}")

                    # Cost estimate
                    estimated_cost = (len(question) + len(response)) / 1000 * 0.003
                    st.caption(f"💰 Estimated cost: ~${estimated_cost:.4f}")

        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("---")
            st.subheader("📜 Recent Questions")

            for i, item in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"🕐 {item['timestamp']} - {item['question'][:60]}..."):
                    st.markdown(f"**Q:** {item['question']}")
                    st.markdown("**A:**")
                    st.markdown(item['response'])

    # =========================================================================
    # TAB 2: DIRECT SQL EXECUTOR
    # =========================================================================
    with tab2:
        st.markdown("""
        Execute SQL queries directly against your trading database.

        **💡 Tips:**
        - Table name is `trades`
        - Common columns: `transaction_datetime`, `product`, `realised_pnl`, `trade_value`, `account_id`
        - Use `LIMIT` to prevent huge results
        """)

        # SQL input with syntax highlighting
        sql_query = st.text_area(
            "SQL Query:",
            value=st.session_state.last_sql,
            height=150,
            help="Enter your SQL query here",
            key="direct_sql"
        )

        col1, col2, col3 = st.columns([2, 2, 3])

        with col1:
            execute_btn = st.button("🔍 Execute Query", type="primary")

        with col2:
            if st.button("📋 Copy From Above"):
                st.session_state.last_sql = sql_query
                st.rerun()

        with col3:
            if st.button("💾 Save to Library"):
                query_name = st.text_input("Query name:", key="save_query_name")
                if query_name:
                    # Save to session state (in real app, would save to database)
                    if 'saved_queries' not in st.session_state:
                        st.session_state.saved_queries = {}
                    st.session_state.saved_queries[query_name] = sql_query
                    st.success(f"Saved '{query_name}'!")

        if execute_btn and sql_query.strip():
            try:
                # Execute query
                result_df = pd.read_sql_query(sql_query, conn)

                # Save to history
                st.session_state.query_history.append({
                    'sql': sql_query,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rows': len(result_df)
                })

                st.session_state.last_results = result_df

                # Display results
                st.success(f"✅ Query executed successfully! {len(result_df)} rows returned")

                # Show first few rows
                st.dataframe(result_df, use_container_width=True, height=400)

                # Download button
                csv = result_df.to_csv(index=False)
                st.download_button(
                    "💾 Download Full Results as CSV",
                    csv,
                    f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key="download_sql_results"
                )

                # Quick stats
                st.markdown("**Quick Stats:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", len(result_df))
                col2.metric("Columns", len(result_df.columns))
                col3.metric("Memory", f"{result_df.memory_usage(deep=True).sum() / 1024:.1f} KB")

            except Exception as e:
                st.error(f"❌ Query Error: {str(e)}")
                st.info("Check your SQL syntax and table/column names")

        # Query history
        if st.session_state.query_history:
            st.markdown("---")
            st.subheader("📜 Query History")

            for i, query in enumerate(reversed(st.session_state.query_history[-10:])):
                with st.expander(f"🕐 {query['timestamp']} - {query['rows']} rows"):
                    st.code(query['sql'], language="sql")
                    if st.button(f"Re-run", key=f"rerun_{i}"):
                        st.session_state.last_sql = query['sql']
                        st.rerun()

    # =========================================================================
    # TAB 3: VISUAL QUERY BUILDER
    # =========================================================================
    with tab3:
        st.markdown("""
        Build queries visually - no SQL knowledge required!
        Select columns, add filters, and Claude will generate the SQL for you.
        """)

        # Available columns from the trades table
        all_columns = {
            'transaction_datetime': 'Date/Time',
            'product': 'Product',
            'realised_pnl': 'Realized P&L',
            'trade_value': 'Trade Value',
            'account_id': 'Account ID',
            'type': 'Transaction Type',
            'amt_per_point': 'Amount per Point',
            'trade_price': 'Trade Price',
            'quantity': 'Quantity',
            'direction': 'Direction',
            'order_id': 'Order ID',
            'trade_id': 'Trade ID',
            'cf_balance': 'Cash Balance'
        }

        st.subheader("1️⃣ Select Columns to Display")

        # Column selection with friendly names
        col1, col2, col3, col4 = st.columns(4)

        selected_columns = []

        with col1:
            if st.checkbox("Date/Time", value=True, key="col_datetime"):
                selected_columns.append('transaction_datetime')
            if st.checkbox("Product", value=True, key="col_product"):
                selected_columns.append('product')
            if st.checkbox("P&L", value=True, key="col_pnl"):
                selected_columns.append('realised_pnl')

        with col2:
            if st.checkbox("Trade Value", value=False, key="col_value"):
                selected_columns.append('trade_value')
            if st.checkbox("Account ID", value=False, key="col_account"):
                selected_columns.append('account_id')
            if st.checkbox("Type", value=False, key="col_type"):
                selected_columns.append('type')

        with col3:
            if st.checkbox("Amt/Point", value=False, key="col_amt"):
                selected_columns.append('amt_per_point')
            if st.checkbox("Trade Price", value=False, key="col_price"):
                selected_columns.append('trade_price')
            if st.checkbox("Quantity", value=False, key="col_qty"):
                selected_columns.append('quantity')

        with col4:
            if st.checkbox("Direction", value=False, key="col_dir"):
                selected_columns.append('direction')
            if st.checkbox("Order ID", value=False, key="col_order"):
                selected_columns.append('order_id')
            if st.checkbox("Cash Balance", value=False, key="col_balance"):
                selected_columns.append('cf_balance')

        if not selected_columns:
            st.warning("⚠️ Please select at least one column")
            selected_columns = ['transaction_datetime', 'product', 'realised_pnl']

        st.markdown("---")
        st.subheader("2️⃣ Add Filters (Optional)")

        # Initialize filters in session state
        if 'visual_filters' not in st.session_state:
            st.session_state.visual_filters = []

        # Add filter button
        if st.button("➕ Add Filter"):
            st.session_state.visual_filters.append({
                'column': 'product',
                'operator': 'equals',
                'value': ''
            })

        # Display existing filters
        filters_to_remove = []
        for i, filter_def in enumerate(st.session_state.visual_filters):
            col1, col2, col3, col4 = st.columns([3, 2, 3, 1])

            with col1:
                filter_column = st.selectbox(
                    "Column",
                    options=list(all_columns.keys()),
                    format_func=lambda x: all_columns[x],
                    key=f"filter_col_{i}",
                    index=list(all_columns.keys()).index(filter_def['column'])
                )

            with col2:
                filter_operator = st.selectbox(
                    "Condition",
                    options=['equals', 'not equals', 'greater than', 'less than', 'contains', 'starts with'],
                    key=f"filter_op_{i}",
                    index=['equals', 'not equals', 'greater than', 'less than', 'contains', 'starts with'].index(filter_def['operator'])
                )

            with col3:
                filter_value = st.text_input(
                    "Value",
                    value=filter_def['value'],
                    key=f"filter_val_{i}"
                )

            with col4:
                if st.button("🗑️", key=f"remove_filter_{i}"):
                    filters_to_remove.append(i)

            # Update filter definition
            st.session_state.visual_filters[i] = {
                'column': filter_column,
                'operator': filter_operator,
                'value': filter_value
            }

        # Remove filters marked for deletion
        for i in reversed(filters_to_remove):
            st.session_state.visual_filters.pop(i)

        st.markdown("---")
        st.subheader("3️⃣ Sort & Limit")

        col1, col2, col3 = st.columns(3)

        with col1:
            sort_column = st.selectbox(
                "Sort by",
                options=selected_columns,
                format_func=lambda x: all_columns.get(x, x),
                key="sort_col"
            )

        with col2:
            sort_order = st.selectbox(
                "Order",
                options=['DESC', 'ASC'],
                format_func=lambda x: 'Newest First' if x == 'DESC' else 'Oldest First',
                key="sort_order"
            )

        with col3:
            limit = st.number_input(
                "Limit rows",
                min_value=10,
                max_value=10000,
                value=100,
                step=10,
                key="limit"
            )

        st.markdown("---")

        # Generate SQL
        def generate_sql(columns, filters, sort_col, sort_order, limit):
            """Generate SQL from visual query builder"""
            # SELECT clause
            sql = f"SELECT {', '.join(columns)}\n"
            sql += "FROM trades\n"

            # WHERE clause
            if filters:
                where_clauses = []
                for f in filters:
                    col = f['column']
                    op = f['operator']
                    val = f['value']

                    if not val:  # Skip empty filters
                        continue

                    if op == 'equals':
                        where_clauses.append(f"{col} = '{val}'")
                    elif op == 'not equals':
                        where_clauses.append(f"{col} != '{val}'")
                    elif op == 'greater than':
                        where_clauses.append(f"{col} > {val}")
                    elif op == 'less than':
                        where_clauses.append(f"{col} < {val}")
                    elif op == 'contains':
                        where_clauses.append(f"{col} LIKE '%{val}%'")
                    elif op == 'starts with':
                        where_clauses.append(f"{col} LIKE '{val}%'")

                if where_clauses:
                    sql += "WHERE " + " AND ".join(where_clauses) + "\n"

            # ORDER BY clause
            sql += f"ORDER BY {sort_col} {sort_order}\n"

            # LIMIT clause
            sql += f"LIMIT {limit};"

            return sql

        generated_sql = generate_sql(
            selected_columns,
            st.session_state.visual_filters,
            sort_column,
            sort_order,
            limit
        )

        st.subheader("4️⃣ Generated SQL")
        st.code(generated_sql, language="sql")

        col1, col2 = st.columns([1, 3])

        with col1:
            run_visual_query = st.button("🚀 Run Query", type="primary", key="run_visual")

        with col2:
            if st.button("📋 Copy to SQL Executor"):
                st.session_state.last_sql = generated_sql
                st.success("✅ Copied to SQL Executor tab!")

        if run_visual_query:
            try:
                result_df = pd.read_sql_query(generated_sql, conn)
                st.success(f"✅ {len(result_df)} rows returned")

                st.dataframe(result_df, use_container_width=True, height=400)

                # Download
                csv = result_df.to_csv(index=False)
                st.download_button(
                    "💾 Download Results",
                    csv,
                    f"visual_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key="download_visual_results"
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # =========================================================================
    # TAB 4: QUERY LIBRARY (Templates)
    # =========================================================================
    with tab4:
        st.markdown("""
        Pre-built queries for common analysis tasks.
        Click any query to load it into the SQL Executor.
        """)

        # Predefined query templates
        query_templates = {
            "Top 10 Profitable Trades": """
SELECT
    transaction_datetime,
    product,
    realised_pnl,
    trade_value,
    account_id
FROM trades
WHERE realised_pnl > 0
ORDER BY realised_pnl DESC
LIMIT 10;
""",
            "Top 10 Losing Trades": """
SELECT
    transaction_datetime,
    product,
    realised_pnl,
    trade_value,
    account_id
FROM trades
WHERE realised_pnl < 0
ORDER BY realised_pnl ASC
LIMIT 10;
""",
            "Monthly P&L Summary": """
SELECT
    strftime('%Y-%m', transaction_datetime) as month,
    COUNT(*) as total_trades,
    SUM(CASE WHEN realised_pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN realised_pnl < 0 THEN 1 ELSE 0 END) as losses,
    ROUND(SUM(realised_pnl), 2) as total_pnl,
    ROUND(AVG(realised_pnl), 2) as avg_pnl
FROM trades
WHERE realised_pnl IS NOT NULL
GROUP BY month
ORDER BY month DESC
LIMIT 12;
""",
            "Product Performance Summary": """
SELECT
    product,
    COUNT(*) as total_trades,
    ROUND(SUM(CASE WHEN realised_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as win_rate,
    ROUND(SUM(realised_pnl), 2) as total_pnl,
    ROUND(AVG(realised_pnl), 2) as avg_pnl,
    ROUND(MAX(realised_pnl), 2) as best_trade,
    ROUND(MIN(realised_pnl), 2) as worst_trade
FROM trades
WHERE realised_pnl IS NOT NULL
GROUP BY product
ORDER BY total_pnl DESC
LIMIT 20;
""",
            "Weekend Trading Activity": """
SELECT
    strftime('%w', transaction_datetime) as day_of_week,
    CASE strftime('%w', transaction_datetime)
        WHEN '0' THEN 'Sunday'
        WHEN '1' THEN 'Monday'
        WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday'
        WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday'
    END as day_name,
    COUNT(*) as total_trades,
    ROUND(SUM(realised_pnl), 2) as total_pnl
FROM trades
WHERE realised_pnl IS NOT NULL
GROUP BY day_of_week
ORDER BY day_of_week;
""",
            "Recent 100 Trades": """
SELECT
    transaction_datetime,
    product,
    type,
    realised_pnl,
    trade_value,
    account_id
FROM trades
ORDER BY transaction_datetime DESC
LIMIT 100;
""",
            "Largest Trades by Value": """
SELECT
    transaction_datetime,
    product,
    trade_value,
    realised_pnl,
    account_id
FROM trades
WHERE trade_value IS NOT NULL
ORDER BY ABS(trade_value) DESC
LIMIT 20;
""",
            "Account Comparison": """
SELECT
    account_id,
    COUNT(*) as total_trades,
    ROUND(SUM(realised_pnl), 2) as total_pnl,
    ROUND(AVG(realised_pnl), 2) as avg_pnl,
    MIN(transaction_datetime) as first_trade,
    MAX(transaction_datetime) as last_trade
FROM trades
WHERE account_id IS NOT NULL
GROUP BY account_id
ORDER BY total_trades DESC;
"""
        }

        # Display templates in cards
        for template_name, template_sql in query_templates.items():
            with st.expander(f"📊 {template_name}"):
                st.code(template_sql, language="sql")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🚀 Run This Query", key=f"run_{template_name}"):
                        try:
                            result_df = pd.read_sql_query(template_sql, conn)
                            st.success(f"✅ {len(result_df)} rows")
                            st.dataframe(result_df, use_container_width=True)

                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                "💾 Download",
                                csv,
                                f"{template_name.lower().replace(' ', '_')}.csv",
                                key=f"dl_{template_name}"
                            )
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                with col2:
                    if st.button(f"📋 Copy to Editor", key=f"copy_{template_name}"):
                        st.session_state.last_sql = template_sql
                        st.success("✅ Copied to SQL Executor!")

        # User saved queries
        if 'saved_queries' in st.session_state and st.session_state.saved_queries:
            st.markdown("---")
            st.subheader("💾 Your Saved Queries")

            for name, sql in st.session_state.saved_queries.items():
                with st.expander(f"⭐ {name}"):
                    st.code(sql, language="sql")
                    if st.button(f"Run", key=f"run_saved_{name}"):
                        st.session_state.last_sql = sql
                        st.rerun()



if __name__ == "__main__":
    main()