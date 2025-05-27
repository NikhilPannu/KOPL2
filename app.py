import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import datetime
import os
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/18ZbDcL0e053w1-IbxX-OflUOs1z07QTcQYUHtLGwl1k/edit?gid=0#gid=0"
WORKSHEET_NAME = "PRD"


# --- Google Sheets Connection ---
@st.cache_data(ttl=3600)
def get_google_sheet_data():
    client = None

    # Try Streamlit secrets first (only if they exist)
    try:
        if hasattr(st, 'secrets') and st.secrets and "gcp_service_account" in st.secrets:
            creds_str = st.secrets["gcp_service_account"]
            if isinstance(creds_str, str):
                creds_dict = json.loads(creds_str)
                client = gspread.service_account_from_dict(creds_dict)
            elif isinstance(creds_str, dict):
                client = gspread.service_account_from_dict(creds_str)
            else:
                st.warning(
                    f"Streamlit secret 'gcp_service_account' is of unexpected type: {type(creds_str)}. Attempting local credentials.")
    except Exception as e:
        st.info(f"Streamlit secrets not available or invalid: {e}. Attempting local credentials.")
        client = None

    # Fallback to local credentials
    if client is None:
        if not os.path.exists("credentials.json"):
            st.error("❌ **Authentication Required**: Please provide either:")
            st.error("1. A 'credentials.json' file in the project directory, OR")
            st.error("2. Configure Streamlit secrets with 'gcp_service_account'")
            st.error("📖 **Setup Instructions**: Check the Google Sheets API documentation for service account setup.")
            st.stop()

        try:
            scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
            local_creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
            client = gspread.authorize(local_creds)
            st.success("✅ Connected using local credentials.json")
        except Exception as e:
            st.error(f"❌ Error loading local 'credentials.json': {e}")
            st.stop()

    try:
        spreadsheet = client.open_by_url(GOOGLE_SHEET_URL)
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        data = worksheet.get_all_values()

        if not data or len(data) < 1:
            st.warning("No data found in the worksheet.")
            return pd.DataFrame()

        headers = data[0]
        records = data[1:]
        df = pd.DataFrame(records, columns=headers)
        df.replace('', np.nan, inplace=True)

        st.success(f"✅ Data loaded successfully: {len(df)} records from '{WORKSHEET_NAME}'")
        return df

    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"❌ Spreadsheet not found. Check the GOOGLE_SHEET_URL: {GOOGLE_SHEET_URL}")
        return pd.DataFrame()
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"❌ Worksheet '{WORKSHEET_NAME}' not found in the spreadsheet.")
        return pd.DataFrame()
    except gspread.exceptions.APIError as api_e:
        st.error(f"❌ Google Sheets API Error: {api_e}")
        return pd.DataFrame()
    except Exception as sheet_error:
        st.error(f"❌ Unexpected error accessing Google Sheet: {sheet_error}")
        return pd.DataFrame()


# --- Enhanced Data Processing ---
@st.cache_data(ttl=3600)
def process_data(df):
    if df.empty:
        return df

    # Date processing
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Time processing
    for col in ['Machine start time', 'Machine End time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce').dt.time

    # Numeric columns processing
    numeric_cols = [
        'Running time', 'Process time (Machining)', 'Process time (Setup)',
        'Mfg qty', 'Rejected qty', 'Approved qty', 'Total Cycle time',
        'Breakdown duration (in minutes)', 'Unreported time'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    # Enhanced KPI calculations
    df['Total Process Time'] = df['Process time (Machining)'] + df['Process time (Setup)']

    # Yield Rate (safe division)
    df['Yield Rate'] = np.where(df['Mfg qty'] > 0,
                                (df['Approved qty'] / df['Mfg qty']) * 100, 0)

    # Rejection Rate
    df['Rejection Rate'] = np.where(df['Mfg qty'] > 0,
                                    (df['Rejected qty'] / df['Mfg qty']) * 100, 0)

    # Machine Utilization (Enhanced)
    total_available_time = df['Running time'] + df['Breakdown duration (in minutes)'] + df['Unreported time']
    df['Machine Utilization'] = np.where(total_available_time > 0,
                                         (df['Running time'] / total_available_time) * 100, 0)

    # Productivity Rate (parts per hour)
    df['Productivity Rate'] = np.where(df['Running time'] > 0,
                                       (df['Approved qty'] / (df['Running time'] / 60)), 0)

    # Fill missing categorical data
    categorical_cols = ['Breakdown (entry)', 'Item code', 'Item name', 'Operator name', 'Shift']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Not Specified')
        else:
            df[col] = 'Not Specified'

    return df


# --- Executive Summary Functions ---
def create_executive_kpis(df):
    """Create executive-level KPIs"""
    total_records = len(df)
    date_range = f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"

    # Key metrics
    total_production = df['Mfg qty'].sum()
    total_approved = df['Approved qty'].sum()
    total_rejected = df['Rejected qty'].sum()
    overall_yield = (total_approved / total_production * 100) if total_production > 0 else 0
    overall_rejection = (total_rejected / total_production * 100) if total_production > 0 else 0

    # Time metrics
    total_breakdown_hours = df['Breakdown duration (in minutes)'].sum() / 60
    total_unreported_hours = df['Unreported time'].sum() / 60
    total_running_hours = df['Running time'].sum() / 60

    # Efficiency metrics
    avg_machine_utilization = df['Machine Utilization'].mean()
    avg_productivity = df['Productivity Rate'].mean()

    return {
        'total_records': total_records,
        'date_range': date_range,
        'total_production': total_production,
        'total_approved': total_approved,
        'total_rejected': total_rejected,
        'overall_yield': overall_yield,
        'overall_rejection': overall_rejection,
        'total_breakdown_hours': total_breakdown_hours,
        'total_unreported_hours': total_unreported_hours,
        'total_running_hours': total_running_hours,
        'avg_machine_utilization': avg_machine_utilization,
        'avg_productivity': avg_productivity
    }


def create_pareto_chart(df, category_col, value_col, title, top_n=10):
    """Create Pareto chart for top issues"""
    grouped = df.groupby(category_col)[value_col].sum().sort_values(ascending=False).head(top_n)
    cumulative_pct = (grouped.cumsum() / grouped.sum() * 100)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart
    fig.add_trace(
        go.Bar(x=grouped.index, y=grouped.values, name=value_col, marker_color='steelblue'),
        secondary_y=False,
    )

    # Cumulative percentage line
    fig.add_trace(
        go.Scatter(x=grouped.index, y=cumulative_pct.values, mode='lines+markers',
                   name='Cumulative %', line=dict(color='red', width=2)),
        secondary_y=True,
    )

    fig.update_xaxes(title_text=category_col)
    fig.update_yaxes(title_text=value_col, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True, range=[0, 100])
    fig.update_layout(title_text=title, showlegend=True)

    return fig


def create_trend_analysis(df, date_col, metrics, title):
    """Create multi-metric trend analysis"""
    daily_data = df.groupby(df[date_col].dt.date).agg({
        metric: 'sum' if metric in ['Breakdown duration (in minutes)', 'Unreported time', 'Rejected qty']
        else 'mean' for metric in metrics
    }).reset_index()

    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=daily_data[date_col],
            y=daily_data[metric],
            mode='lines+markers',
            name=metric,
            line=dict(color=colors[i % len(colors)])
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        showlegend=True,
        height=400
    )

    return fig


# --- Main Dashboard ---
st.set_page_config(layout="wide", page_title="Manufacturing Executive Dashboard", page_icon="🏭")

# Header
st.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;'>
    <h1>🏭 Manufacturing Executive Dashboard</h1>
    <p>Real-time Production Performance Analytics</p>
</div>
""", unsafe_allow_html=True)

# Load and process data
with st.spinner("🔄 Loading data from Google Sheets..."):
    df_raw = get_google_sheet_data()

if df_raw.empty:
    st.stop()

df_processed = process_data(df_raw.copy())
df_processed = df_processed.sort_values(by='Date').reset_index(drop=True)

# Sidebar Filters
st.sidebar.markdown("## 🎛️ Filters")

# Date filter
if not df_processed['Date'].isna().all():
    min_date = df_processed['Date'].min().date()
    max_date = df_processed['Date'].max().date()

    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df_processed[
            (df_processed['Date'].dt.date >= start_date) &
            (df_processed['Date'].dt.date <= end_date)
            ]
    else:
        df_filtered = df_processed
else:
    df_filtered = df_processed
    st.sidebar.warning("No valid dates found in data")

# Additional filters
machines = ['All'] + sorted(df_filtered['Machine number'].astype(str).unique().tolist())
selected_machine = st.sidebar.selectbox("🔧 Machine", machines)
if selected_machine != 'All':
    df_filtered = df_filtered[df_filtered['Machine number'].astype(str) == selected_machine]

shifts = ['All'] + sorted(df_filtered['Shift'].astype(str).unique().tolist())
selected_shift = st.sidebar.selectbox("⏰ Shift", shifts)
if selected_shift != 'All':
    df_filtered = df_filtered[df_filtered['Shift'].astype(str) == selected_shift]

# Main Dashboard
if df_filtered.empty:
    st.error("❌ No data available for selected filters")
    st.stop()

# Executive Summary
st.markdown("## 📊 Executive Summary")
kpis = create_executive_kpis(df_filtered)

# KPI Cards
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Production", f"{int(kpis['total_production']):,}",
              help="Total manufactured quantity")

with col2:
    st.metric("Yield Rate", f"{kpis['overall_yield']:.1f}%",
              delta=f"{kpis['overall_yield'] - 95:.1f}%" if kpis['overall_yield'] < 95 else None,
              help="Overall production yield")

with col3:
    st.metric("Rejection Rate", f"{kpis['overall_rejection']:.1f}%",
              delta=f"{kpis['overall_rejection'] - 5:.1f}%" if kpis['overall_rejection'] > 5 else None,
              delta_color="inverse",
              help="Overall rejection rate")

with col4:
    st.metric("Machine Utilization", f"{kpis['avg_machine_utilization']:.1f}%",
              help="Average machine utilization")

with col5:
    st.metric("Breakdown Hours", f"{kpis['total_breakdown_hours']:.0f}h",
              help="Total breakdown time")

with col6:
    st.metric("Unreported Hours", f"{kpis['total_unreported_hours']:.0f}h",
              help="Total unreported time")

st.markdown("---")

# Critical Issues Analysis
st.markdown("## 🚨 Critical Issues Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔧 Top Breakdown Reasons (Pareto Analysis)")
    if df_filtered['Breakdown duration (in minutes)'].sum() > 0:
        pareto_fig = create_pareto_chart(
            df_filtered,
            'Breakdown (entry)',
            'Breakdown duration (in minutes)',
            'Breakdown Analysis - Focus on Top Issues'
        )
        st.plotly_chart(pareto_fig, use_container_width=True)
    else:
        st.info("No breakdown data available")

with col2:
    st.markdown("### ⏱️ Unreported Time by Machine")
    unreported_machine = df_filtered.groupby('Machine number')['Unreported time'].sum().sort_values(
        ascending=False).head(10)
    if unreported_machine.sum() > 0:
        fig = px.bar(x=unreported_machine.index, y=unreported_machine.values,
                     title='Machines with Highest Unreported Time',
                     labels={'x': 'Machine Number', 'y': 'Unreported Time (minutes)'},
                     color=unreported_machine.values,
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No unreported time data available")

# Quality Issues
st.markdown("### 🎯 Quality Issues Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Rejection Rate by Item")
    rejection_by_item = df_filtered.groupby('Item code').agg({
        'Mfg qty': 'sum',
        'Rejected qty': 'sum'
    })
    rejection_by_item['Rejection Rate'] = (
                rejection_by_item['Rejected qty'] / rejection_by_item['Mfg qty'] * 100).fillna(0)
    rejection_by_item = rejection_by_item.sort_values('Rejection Rate', ascending=False).head(10)

    if not rejection_by_item.empty:
        fig = px.bar(x=rejection_by_item.index, y=rejection_by_item['Rejection Rate'],
                     title='Items with Highest Rejection Rates',
                     labels={'x': 'Item Code', 'y': 'Rejection Rate (%)'},
                     color=rejection_by_item['Rejection Rate'],
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rejection data available")

with col2:
    st.markdown("#### Production Efficiency by Operator")
    operator_performance = df_filtered.groupby('Operator name').agg({
        'Approved qty': 'sum',
        'Rejected qty': 'sum',
        'Running time': 'sum'
    })
    operator_performance['Efficiency'] = (operator_performance['Approved qty'] /
                                          (operator_performance['Running time'] / 60)).fillna(0)
    operator_performance = operator_performance.sort_values('Efficiency', ascending=False).head(10)

    if not operator_performance.empty:
        fig = px.bar(x=operator_performance.index, y=operator_performance['Efficiency'],
                     title='Operator Efficiency (Parts/Hour)',
                     labels={'x': 'Operator', 'y': 'Parts per Hour'},
                     color=operator_performance['Efficiency'],
                     color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No operator performance data available")

# Trend Analysis
st.markdown("## 📈 Trend Analysis")

# Multi-metric trends
if len(df_filtered) > 1:
    trend_fig = create_trend_analysis(
        df_filtered,
        'Date',
        ['Breakdown duration (in minutes)', 'Unreported time', 'Rejection Rate', 'Yield Rate'],
        'Key Metrics Trend Analysis'
    )
    st.plotly_chart(trend_fig, use_container_width=True)

# Machine Performance Heatmap
st.markdown("### 🔥 Machine Performance Heatmap")
if len(df_filtered) > 0:
    machine_daily = df_filtered.groupby(['Machine number', df_filtered['Date'].dt.date]).agg({
        'Machine Utilization': 'mean',
        'Breakdown duration (in minutes)': 'sum',
        'Yield Rate': 'mean'
    }).reset_index()

    if not machine_daily.empty:
        # Create heatmap for machine utilization
        pivot_util = machine_daily.pivot(index='Machine number', columns='Date', values='Machine Utilization')
        fig = px.imshow(pivot_util,
                        title='Machine Utilization Heatmap (%)',
                        color_continuous_scale='RdYlGn',
                        aspect='auto')
        st.plotly_chart(fig, use_container_width=True)

# Action Items Table
st.markdown("## 🎯 Priority Action Items")

# Generate action items based on data
action_items = []

# Critical breakdown reasons
top_breakdowns = df_filtered.groupby('Breakdown (entry)')['Breakdown duration (in minutes)'].sum().sort_values(
    ascending=False).head(3)
for reason, duration in top_breakdowns.items():
    if duration > 0:
        action_items.append({
            'Priority': '🔴 High',
            'Category': 'Breakdown',
            'Issue': f'{reason}',
            'Impact': f'{duration:.0f} minutes lost',
            'Recommended Action': 'Investigate root cause and implement preventive measures'
        })

# High rejection items
high_rejection_items = rejection_by_item.head(3)
for item, row in high_rejection_items.iterrows():
    if row['Rejection Rate'] > 5:
        action_items.append({
            'Priority': '🟡 Medium',
            'Category': 'Quality',
            'Issue': f'High rejection rate for {item}',
            'Impact': f'{row["Rejection Rate"]:.1f}% rejection rate',
            'Recommended Action': 'Review process parameters and quality controls'
        })

# High unreported time machines
high_unreported = df_filtered.groupby('Machine number')['Unreported time'].sum().sort_values(ascending=False).head(3)
for machine, time in high_unreported.items():
    if time > 60:  # More than 1 hour
        action_items.append({
            'Priority': '🟠 Medium',
            'Category': 'Efficiency',
            'Issue': f'High unreported time for Machine {machine}',
            'Impact': f'{time:.0f} minutes unreported',
            'Recommended Action': 'Improve time tracking and operator training'
        })

if action_items:
    action_df = pd.DataFrame(action_items)
    st.dataframe(action_df, use_container_width=True)
else:
    st.success("✅ No critical issues identified in current data")

# Data Export
st.markdown("## 📥 Data Export")
col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Download Summary Report"):
        summary_data = {
            'Date Range': kpis['date_range'],
            'Total Production': kpis['total_production'],
            'Yield Rate': f"{kpis['overall_yield']:.2f}%",
            'Rejection Rate': f"{kpis['overall_rejection']:.2f}%",
            'Breakdown Hours': f"{kpis['total_breakdown_hours']:.1f}",
            'Unreported Hours': f"{kpis['total_unreported_hours']:.1f}",
            'Machine Utilization': f"{kpis['avg_machine_utilization']:.1f}%"
        }
        st.json(summary_data)

with col2:
    if st.button("📋 Download Filtered Data"):
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name=f"manufacturing_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("*Dashboard last updated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")