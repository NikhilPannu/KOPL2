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
WORKSHEET_NAMES_MAP = {
    "prd": "PRD",
    "breakdown_form": "Form Responses 1"
}


# --- Google Sheets Connection ---
@st.cache_data(ttl=3600)
def get_google_sheets_data(sheet_url, sheet_names_map):
    client = None
    try:
        if hasattr(st, 'secrets') and st.secrets and "gcp_service_account" in st.secrets:
            creds_str = st.secrets["gcp_service_account"]
            if isinstance(creds_str, str):
                creds_dict = json.loads(creds_str)
            elif isinstance(creds_str, dict):
                creds_dict = creds_str
            else:
                st.warning(
                    f"Streamlit secret 'gcp_service_account' is of unexpected type: {type(creds_str)}. Attempting local credentials.")
                creds_dict = None  # Fallback
            if creds_dict:
                client = gspread.service_account_from_dict(creds_dict)
    except Exception as e:
        st.info(f"Streamlit secrets not available or invalid: {e}. Attempting local credentials.")
        client = None

    if client is None:
        if not os.path.exists("credentials.json"):
            st.error(
                "❌ **Authentication Required**: Provide 'credentials.json' or configure Streamlit secrets 'gcp_service_account'.")
            st.stop()
        try:
            scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
            local_creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
            client = gspread.authorize(local_creds)
            st.success("✅ Connected using local credentials.json")
        except Exception as e:
            st.error(f"❌ Error loading local 'credentials.json': {e}")
            st.stop()

    dataframes = {}
    all_sheets_loaded_successfully = True
    try:
        spreadsheet = client.open_by_url(sheet_url)
        for key, sheet_name in sheet_names_map.items():
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
                data = worksheet.get_all_values()
                if not data or len(data) < 1:
                    st.warning(f"No data found in worksheet '{sheet_name}'. An empty DataFrame will be used.")
                    dataframes[key] = pd.DataFrame()
                    continue

                headers = data[0]
                records = data[1:]
                df = pd.DataFrame(records, columns=headers)
                df.replace('', np.nan, inplace=True)  # Replace all empty strings with NaN
                dataframes[key] = df
                st.success(f"✅ Data loaded: {len(df)} records from '{sheet_name}'")
            except gspread.exceptions.WorksheetNotFound:
                st.error(f"❌ Worksheet '{sheet_name}' not found. An empty DataFrame will be used for this sheet.")
                dataframes[key] = pd.DataFrame()
                if key == "prd":  # PRD sheet is critical
                    all_sheets_loaded_successfully = False
            except Exception as ws_e:
                st.error(f"❌ Error loading worksheet '{sheet_name}': {ws_e}")
                dataframes[key] = pd.DataFrame()
                if key == "prd":
                    all_sheets_loaded_successfully = False
        return dataframes, all_sheets_loaded_successfully

    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"❌ Spreadsheet not found. Check GOOGLE_SHEET_URL: {sheet_url}")
        return {key: pd.DataFrame() for key in sheet_names_map}, False
    except gspread.exceptions.APIError as api_e:
        st.error(f"❌ Google Sheets API Error: {api_e}")
        return {key: pd.DataFrame() for key in sheet_names_map}, False
    except Exception as sheet_error:
        st.error(f"❌ Unexpected error accessing Google Sheet: {sheet_error}")
        return {key: pd.DataFrame() for key in sheet_names_map}, False


# --- Enhanced Data Processing ---
@st.cache_data(ttl=3600)
def process_data(df_prd_raw, df_breakdown_form_raw):
    if df_prd_raw.empty:
        st.error("PRD data is empty. Cannot proceed with processing.")
        return pd.DataFrame()

    df_prd = df_prd_raw.copy()
    df_breakdown_form = df_breakdown_form_raw.copy()

    st.write("Initial PRD columns:", df_prd.columns.tolist())
    if not df_breakdown_form.empty:
        st.write("Initial Breakdown Form columns:", df_breakdown_form.columns.tolist())

    # --- Process PRD Data ---
    # Ensure Record ID is string and handle potential NaN before type conversion
    if 'Record ID' in df_prd.columns:
        df_prd['Record ID'] = df_prd['Record ID'].astype(str).fillna('Unknown_Record_ID')
    else:
        st.error("'Record ID' column is missing in PRD sheet. This is crucial for merging.")
        df_prd['Record ID'] = 'Unknown_Record_ID_Placeholder'  # Add placeholder to avoid crashing

    df_prd['Date'] = pd.to_datetime(df_prd['Date'], errors='coerce', dayfirst=True)

    time_cols_prd = ['Machine start time', 'Machine End time']
    for col in time_cols_prd:
        if col in df_prd.columns:
            df_prd[col] = pd.to_datetime(df_prd[col], format='%H:%M:%S', errors='coerce').dt.time
        else:
            df_prd[col] = None  # or pd.NaT based on desired handling

    # Store original PRD breakdown info before it's potentially modified by mappings or merged data
    if 'Breakdown duration (in minutes)' in df_prd.columns:
        df_prd['PRD_Original_Breakdown_Duration'] = pd.to_numeric(df_prd['Breakdown duration (in minutes)'],
                                                                  errors='coerce').fillna(0)
    else:
        df_prd['PRD_Original_Breakdown_Duration'] = 0

    if 'Breakdown (entry)' in df_prd.columns:
        df_prd['PRD_Original_Breakdown_Entry'] = df_prd['Breakdown (entry)'].astype(str).fillna('Not Specified')
    else:
        df_prd['PRD_Original_Breakdown_Entry'] = 'Not Specified'

    numeric_columns_mapping_prd = {
        # Standard Name : [Possible Original Names in Sheet]
        'Running time': ['Running time', 'running time', 'Machine running time'],
        'Process time (Machining)': ['Process time (Machining)', 'Machining time', 'Process time'],
        'Process time (Setup)': ['Process time (Setup)', 'Setup time'],
        'Mfg qty': ['Mfg qty', 'Manufacturing qty', 'Manufactured qty'],
        'Rejected qty': ['Rejected qty', 'Reject qty', 'Rejection qty'],
        'Approved qty': ['Approved qty', 'Good qty', 'Accepted qty'],
        'Total Cycle time': ['Total Cycle time', 'Cycle time'],
        'Unreported time': ['Unreported time', 'Unaccounted time'],
        # PRD's own breakdown duration is handled by PRD_Original_Breakdown_Duration
    }
    for standard_col, possible_cols in numeric_columns_mapping_prd.items():
        found_col = None
        for col_variant in possible_cols:
            if col_variant in df_prd.columns:
                found_col = col_variant
                break
        if found_col:
            df_prd[standard_col] = pd.to_numeric(df_prd[found_col], errors='coerce').fillna(0)
        else:
            df_prd[standard_col] = 0
            if standard_col in ['Running time']:  # Critical for many calcs
                st.warning(
                    f"Critical numeric column '{standard_col}' (or its variants) not found in PRD sheet. Defaulting to 0.")

    categorical_columns_mapping_prd = {
        'Machine number': ['Machine number', 'Machine', 'Machine ID'],
        'Shift': ['Shift', 'Shift name'],
        'BMR Number': ['BMR Number', 'BMR No'],
        'Item code': ['Item code', 'Product code', 'Part code'],
        'Item name': ['Item name', 'Product name', 'Part name'],
        'Operation or Process description': ['Operation or Process description', 'Process Description',
                                             'Operation Description'],
        'Operator name': ['Operator name', 'Operator', 'Worker name'],
        'verified by': ['verified by', 'Verified By', 'Checked By'],
        'Report Breakdown': ['Report Breakdown', 'Breakdown Reported in PRD', 'PRD Breakdown'],
        # e.g. Yes/No if breakdown noted in PRD
        # PRD's own breakdown entry is handled by PRD_Original_Breakdown_Entry
    }
    for standard_col, possible_cols in categorical_columns_mapping_prd.items():
        found_col = None
        for col_variant in possible_cols:
            if col_variant in df_prd.columns:
                found_col = col_variant
                break
        if found_col:
            df_prd[standard_col] = df_prd[found_col].astype(str).fillna('Not Specified')
            if standard_col != found_col:  # Rename to standard if different
                df_prd.drop(columns=[found_col], inplace=True, errors='ignore')
        else:
            df_prd[standard_col] = 'Not Specified'
            # st.warning(f"Categorical column '{standard_col}' (or its variants) not found in PRD sheet. Defaulting to 'Not Specified'.")

    # Handle 'Remark (if any)' separately as it can be blank
    if 'Remark (if any)' in df_prd.columns:
        df_prd['Remark (if any)'] = df_prd['Remark (if any)'].astype(str).fillna('')
    else:
        df_prd['Remark (if any)'] = ''

    # --- Process Breakdown Form Data ---
    df_breakdown_agg = pd.DataFrame()  # Initialize empty
    if not df_breakdown_form.empty and 'Record ID' in df_breakdown_form.columns and 'Breakdown START TIME' in df_breakdown_form.columns and 'Breakdown END TIME' in df_breakdown_form.columns:
        df_bd_form = df_breakdown_form.copy()
        df_bd_form.rename(columns={  # Standardize form column names
            'Timestamp': 'Form_Timestamp',
            'Record ID': 'Form_Record_ID',
            'Breakdown START TIME': 'Form_BD_Start',
            'Breakdown END TIME': 'Form_BD_End',
            'Breakdown REASON': 'Form_BD_Reason',
            'Breakdown REPORTED BY': 'Form_BD_Reported_By'
        }, inplace=True)

        df_bd_form['Form_Record_ID'] = df_bd_form['Form_Record_ID'].astype(str).fillna('Unknown_Form_Record_ID')
        df_bd_form['Form_Timestamp'] = pd.to_datetime(df_bd_form['Form_Timestamp'], errors='coerce')

        # Merge with PRD to get the Date for breakdown time calculations
        # Ensure 'Date' from df_prd is available and valid before merging
        if 'Date' not in df_prd.columns or df_prd['Date'].isnull().all():
            st.warning(
                "PRD 'Date' column is missing or all null. Cannot accurately calculate breakdown durations from form.")
        else:
            # Only merge necessary columns to avoid conflicts, PRD 'Date' is crucial.
            df_bd_form_dated = pd.merge(df_bd_form,
                                        df_prd[['Record ID', 'Date']].rename(columns={'Record ID': 'Form_Record_ID'}),
                                        on='Form_Record_ID',
                                        how='left')
            df_bd_form_dated.dropna(subset=['Date'], inplace=True)  # Drop if no matching PRD date

            if not df_bd_form_dated.empty:
                # Helper to combine date and time string/object
                def combine_date_time(s_date, s_time_str):
                    s_time_str = s_time_str.astype(str)
                    # Attempt to parse time, handling potential AM/PM or 24hr format issues robustly
                    # Common formats: HH:MM:SS, HH:MM, H:MM AM/PM
                    return pd.to_datetime(s_date.dt.strftime('%Y-%m-%d') + ' ' + s_time_str, errors='coerce')

                df_bd_form_dated['Form_BD_Start_dt'] = combine_date_time(df_bd_form_dated['Date'],
                                                                         df_bd_form_dated['Form_BD_Start'])
                df_bd_form_dated['Form_BD_End_dt'] = combine_date_time(df_bd_form_dated['Date'],
                                                                       df_bd_form_dated['Form_BD_End'])

                # Handle end times past midnight
                mask_midnight_span = (df_bd_form_dated['Form_BD_End_dt'].notna()) & \
                                     (df_bd_form_dated['Form_BD_Start_dt'].notna()) & \
                                     (df_bd_form_dated['Form_BD_End_dt'] < df_bd_form_dated['Form_BD_Start_dt'])
                df_bd_form_dated.loc[mask_midnight_span, 'Form_BD_End_dt'] += pd.Timedelta(days=1)

                df_bd_form_dated['Form_BD_Duration_Minutes_Calc'] = (df_bd_form_dated['Form_BD_End_dt'] -
                                                                     df_bd_form_dated[
                                                                         'Form_BD_Start_dt']).dt.total_seconds() / 60
                df_bd_form_dated['Form_BD_Duration_Minutes_Calc'] = df_bd_form_dated[
                    'Form_BD_Duration_Minutes_Calc'].apply(lambda x: max(0, x) if pd.notnull(x) else 0)

                # Aggregate form breakdown data by Record ID
                df_breakdown_agg = df_bd_form_dated.groupby('Form_Record_ID').agg(
                    Total_Form_Breakdown_Duration_Minutes=('Form_BD_Duration_Minutes_Calc', 'sum'),
                    Form_Breakdown_Reasons=('Form_BD_Reason', lambda x: '; '.join(
                        x.dropna().astype(str).unique()) if not x.dropna().empty else 'Not Specified'),
                    Form_Breakdown_Reported_By=('Form_BD_Reported_By', lambda x: '; '.join(
                        x.dropna().astype(str).unique()) if not x.dropna().empty else 'Not Specified'),
                    Form_Breakdown_Count=('Form_BD_Reason', 'count')
                ).reset_index()
    else:
        st.info(
            "Breakdown form data is empty or missing key columns ('Record ID', 'Breakdown START TIME', 'Breakdown END TIME'). Using PRD breakdown data if available.")

    # --- Merge PRD with Aggregated Breakdown Form Data ---
    if not df_breakdown_agg.empty:
        df_merged = pd.merge(df_prd, df_breakdown_agg, left_on='Record ID', right_on='Form_Record_ID', how='left')
        if 'Form_Record_ID' in df_merged.columns:  # Drop redundant key column
            df_merged.drop(columns=['Form_Record_ID'], inplace=True)
    else:
        df_merged = df_prd.copy()  # No form data to merge
        # Ensure columns that would come from form_agg are present if expected later
        df_merged['Total_Form_Breakdown_Duration_Minutes'] = 0
        df_merged['Form_Breakdown_Reasons'] = 'Not Specified'
        df_merged['Form_Breakdown_Reported_By'] = 'Not Specified'
        df_merged['Form_Breakdown_Count'] = 0

    # Fill NaNs for columns that came from the merge (if any rows in PRD didn't have a match in form)
    df_merged['Total_Form_Breakdown_Duration_Minutes'].fillna(0, inplace=True)
    df_merged['Form_Breakdown_Reasons'].fillna('Not Specified', inplace=True)
    df_merged['Form_Breakdown_Reported_By'].fillna('Not Specified', inplace=True)
    df_merged['Form_Breakdown_Count'].fillna(0, inplace=True)

    # --- Consolidate Breakdown Information ---
    # Prioritize form data. If no form data (Form_Breakdown_Count == 0), use PRD's original breakdown data.
    df_merged['Breakdown duration (in minutes)'] = np.where(
        df_merged['Form_Breakdown_Count'] > 0,
        df_merged['Total_Form_Breakdown_Duration_Minutes'],
        df_merged['PRD_Original_Breakdown_Duration']
    )
    df_merged['Breakdown (entry)'] = np.where(
        df_merged['Form_Breakdown_Count'] > 0,
        df_merged['Form_Breakdown_Reasons'],
        df_merged['PRD_Original_Breakdown_Entry']
    )
    # Ensure these are correct type after np.where
    df_merged['Breakdown duration (in minutes)'] = pd.to_numeric(df_merged['Breakdown duration (in minutes)'],
                                                                 errors='coerce').fillna(0)
    df_merged['Breakdown (entry)'] = df_merged['Breakdown (entry)'].astype(str).fillna('Not Specified')

    # --- Final KPI Calculations ---
    df_merged['Total Process Time'] = df_merged['Process time (Machining)'] + df_merged['Process time (Setup)']
    df_merged['Yield Rate'] = np.where(df_merged['Mfg qty'] > 0,
                                       (df_merged['Approved qty'] / df_merged['Mfg qty']) * 100, 0)
    df_merged['Rejection Rate'] = np.where(df_merged['Mfg qty'] > 0,
                                           (df_merged['Rejected qty'] / df_merged['Mfg qty']) * 100, 0)

    # Machine Utilization: Ensure all components are numeric
    running_time = pd.to_numeric(df_merged['Running time'], errors='coerce').fillna(0)
    breakdown_duration = pd.to_numeric(df_merged['Breakdown duration (in minutes)'], errors='coerce').fillna(0)
    unreported_time = pd.to_numeric(df_merged['Unreported time'], errors='coerce').fillna(0)
    total_available_time = running_time + breakdown_duration + unreported_time
    df_merged['Machine Utilization'] = np.where(total_available_time > 0, (running_time / total_available_time) * 100,
                                                0)

    df_merged['Productivity Rate'] = np.where(running_time > 0, (df_merged['Approved qty'] / (running_time / 60)),
                                              0)  # Parts per hour

    # Flag for missing breakdown form entries if PRD indicated a breakdown
    if 'Report Breakdown' in df_merged.columns:  # Check if column exists
        df_merged['Missing_Form_Entry'] = (
                df_merged['Report Breakdown'].str.lower().isin(['yes', 'y', 'true']) & (
                    df_merged['Form_Breakdown_Count'] == 0)
        )
    else:
        df_merged['Missing_Form_Entry'] = False

    st.write("Processed columns:", df_merged.columns.tolist())
    # st.dataframe(df_merged.head()) # For debugging processed data

    return df_merged


# --- Executive Summary Functions (largely unchanged, but will use processed data) ---
def create_executive_kpis(df):
    """Create executive-level KPIs"""
    total_records = len(df)

    valid_dates = df['Date'].dropna()
    date_range = "No valid dates"
    if not valid_dates.empty:
        min_d, max_d = valid_dates.min(), valid_dates.max()
        if pd.notna(min_d) and pd.notna(max_d):
            date_range = f"{min_d.strftime('%Y-%m-%d')} to {max_d.strftime('%Y-%m-%d')}"

    # Key metrics
    total_production = df['Mfg qty'].sum()
    total_approved = df['Approved qty'].sum()
    total_rejected = df['Rejected qty'].sum()
    overall_yield = (total_approved / total_production * 100) if total_production > 0 else 0
    overall_rejection = (total_rejected / total_production * 100) if total_production > 0 else 0

    # Time metrics
    total_breakdown_hours = df['Breakdown duration (in minutes)'].sum() / 60
    total_unreported_hours = df['Unreported time'].sum() / 60  # Assuming 'Unreported time' is in minutes
    total_running_hours = df['Running time'].sum() / 60  # Assuming 'Running time' is in minutes

    # Efficiency metrics
    avg_machine_utilization = df['Machine Utilization'].mean() if not df['Machine Utilization'].empty else 0
    avg_productivity = df['Productivity Rate'].mean() if not df['Productivity Rate'].empty else 0

    # New KPI: Missing Form Entries
    missing_form_entries_count = df['Missing_Form_Entry'].sum() if 'Missing_Form_Entry' in df.columns else 0

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
        'avg_productivity': avg_productivity,
        'missing_form_entries_count': missing_form_entries_count
    }


def create_pareto_chart(df, category_col, value_col, title, top_n=10):
    if category_col not in df.columns or value_col not in df.columns:
        st.warning(f"Pareto chart: Missing '{category_col}' or '{value_col}'.")
        return None
    if df[value_col].sum() == 0:  # No data to plot
        # st.info(f"Pareto chart: No data for '{value_col}'.")
        return None

    grouped = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    grouped = grouped[grouped > 0].head(top_n)  # Only positive values and top N

    if grouped.empty:
        return None

    cumulative_pct = (grouped.cumsum() / grouped.sum() * 100)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=grouped.index, y=grouped.values, name=value_col, marker_color='steelblue'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=grouped.index, y=cumulative_pct.values, mode='lines+markers', name='Cumulative %',
                             line=dict(color='red', width=2)), secondary_y=True)
    fig.update_xaxes(title_text=category_col)
    fig.update_yaxes(title_text=value_col, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True, range=[0, 100.5])  # Ensure 100% is visible
    fig.update_layout(title_text=title, showlegend=True)
    return fig


def create_trend_analysis(df, date_col, metrics, title):
    if date_col not in df.columns:
        st.warning(f"Trend analysis: Missing date column '{date_col}'.")
        return None

    valid_dates_df = df[df[date_col].notna()]
    if valid_dates_df.empty:
        return None

    # Ensure metrics exist in df
    valid_metrics = [m for m in metrics if m in df.columns]
    if not valid_metrics:
        st.warning("Trend analysis: None of the specified metrics found in data.")
        return None

    agg_dict = {
        metric: 'sum' if metric in ['Breakdown duration (in minutes)', 'Unreported time', 'Rejected qty', 'Mfg qty',
                                    'Approved qty']
        else 'mean' for metric in valid_metrics
    }

    # Group by date part of datetime column
    daily_data = valid_dates_df.groupby(valid_dates_df[date_col].dt.date).agg(agg_dict).reset_index()
    daily_data.rename(columns={date_col: 'Date_Axis'}, inplace=True)  # Rename to avoid conflict if 'Date' is a metric

    if daily_data.empty:
        return None

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    for i, metric in enumerate(valid_metrics):
        if metric in daily_data.columns:
            fig.add_trace(go.Scatter(
                x=daily_data['Date_Axis'],
                y=daily_data[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[i % len(colors)])
            ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Value", showlegend=True, height=400)
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
    all_data_dict, sheets_loaded_status = get_google_sheets_data(GOOGLE_SHEET_URL, WORKSHEET_NAMES_MAP)

if not sheets_loaded_status or all_data_dict["prd"].empty:
    st.error("❌ Critical PRD data could not be loaded. Dashboard cannot proceed.")
    st.stop()

df_prd_raw = all_data_dict.get("prd", pd.DataFrame())
df_breakdown_form_raw = all_data_dict.get("breakdown_form", pd.DataFrame())

with st.spinner("⚙️ Processing data... This may take a moment."):
    df_processed = process_data(df_prd_raw, df_breakdown_form_raw)

if df_processed.empty:
    st.error("❌ Data processing failed or resulted in an empty dataset.")
    st.stop()

# Sort by date for trend analyses
df_processed = df_processed.sort_values(by='Date', ascending=True).reset_index(drop=True)

# Sidebar Filters
st.sidebar.markdown("## 🎛️ Filters")

# Date filter
valid_dates_for_filter = df_processed['Date'].dropna()
if not valid_dates_for_filter.empty:
    min_date_filter = valid_dates_for_filter.min().date()
    max_date_filter = valid_dates_for_filter.max().date()
    if min_date_filter > max_date_filter:  # Should not happen if data is sorted
        min_date_filter, max_date_filter = max_date_filter, min_date_filter  # Swap

    date_range_selected = st.sidebar.date_input(
        "📅 Select Date Range",
        value=(min_date_filter, max_date_filter),
        min_value=min_date_filter,
        max_value=max_date_filter,
        key="date_filter"
    )
    if len(date_range_selected) == 2:
        start_date_filter, end_date_filter = date_range_selected
        # Ensure start_date is not after end_date from picker
        if start_date_filter > end_date_filter:
            start_date_filter, end_date_filter = end_date_filter, start_date_filter  # Swap

        df_filtered = df_processed[
            (df_processed['Date'].dt.date >= start_date_filter) &
            (df_processed['Date'].dt.date <= end_date_filter)
            ]
    else:  # Should be 2, but as a fallback
        df_filtered = df_processed.copy()
else:
    df_filtered = df_processed.copy()
    st.sidebar.warning("No valid dates found in data for filtering.")

# Additional filters (ensure columns exist before creating selectbox)
filter_columns = {
    'Machine number': '🔧 Machine',
    'Shift': '⏰ Shift',
    'Item code': '📦 Item Code',
    'Operator name': '👨‍🔧 Operator Name',
    'BMR Number': '📄 BMR Number'
}

for col_name, display_name in filter_columns.items():
    if col_name in df_filtered.columns:
        unique_values = ['All'] + sorted(df_filtered[col_name].astype(str).unique().tolist())
        selected_value = st.sidebar.selectbox(display_name, unique_values, key=f"filter_{col_name}")
        if selected_value != 'All':
            df_filtered = df_filtered[df_filtered[col_name].astype(str) == selected_value]
    else:
        st.sidebar.caption(f"'{col_name}' column not found for filtering.")

# Main Dashboard
if df_filtered.empty:
    st.error("❌ No data available for the selected filters. Please adjust your filter criteria.")
    st.stop()

# Executive Summary
st.markdown("## 📊 Executive Summary")
kpis = create_executive_kpis(df_filtered)

# KPI Cards
cols_kpi = st.columns(7)  # Adjusted for new KPI

with cols_kpi[0]:
    st.metric("Total Production", f"{int(kpis['total_production']):,}", help="Total manufactured quantity")
with cols_kpi[1]:
    st.metric("Yield Rate", f"{kpis['overall_yield']:.1f}%",
              delta=f"{kpis['overall_yield'] - 95:.1f}%" if kpis['overall_yield'] < 95 else None,
              help="Overall production yield (Approved Qty / Mfg Qty)")
with cols_kpi[2]:
    st.metric("Rejection Rate", f"{kpis['overall_rejection']:.1f}%",
              delta=f"{kpis['overall_rejection'] - 5:.1f}%" if kpis['overall_rejection'] > 5 else None,
              delta_color="inverse", help="Overall rejection rate (Rejected Qty / Mfg Qty)")
with cols_kpi[3]:
    st.metric("Machine Utilization", f"{kpis['avg_machine_utilization']:.1f}%",
              help="Avg. Machine Utilization (Running Time / (Run Time + Breakdown + Unreported))")
with cols_kpi[4]:
    st.metric("Breakdown Hours", f"{kpis['total_breakdown_hours']:.1f}h", help="Total breakdown time in hours")
with cols_kpi[5]:
    st.metric("Unreported Hours", f"{kpis['total_unreported_hours']:.1f}h", help="Total unreported time in hours")
with cols_kpi[6]:
    st.metric("Missing BD Forms", f"{int(kpis['missing_form_entries_count'])}",
              help="Count of PRD records indicating breakdown but missing a form entry")

st.markdown("---")

# Critical Issues Analysis
st.markdown("## 🚨 Critical Issues Analysis")
col1_crit, col2_crit = st.columns(2)

with col1_crit:
    st.markdown("### 🔧 Top Breakdown Reasons (Pareto)")
    if 'Breakdown duration (in minutes)' in df_filtered.columns and 'Breakdown (entry)' in df_filtered.columns:
        pareto_fig_bd = create_pareto_chart(
            df_filtered[df_filtered['Breakdown (entry)'] != 'Not Specified'],  # Exclude 'Not Specified' from Pareto
            'Breakdown (entry)',
            'Breakdown duration (in minutes)',
            'Top Breakdown Reasons by Duration'
        )
        if pareto_fig_bd:
            st.plotly_chart(pareto_fig_bd, use_container_width=True)
        else:
            st.info("No breakdown data with specified reasons to display for Pareto chart.")
    else:
        st.info("Breakdown columns not available for Pareto chart.")

with col2_crit:
    st.markdown("### ⏱️ Unreported Time by Machine")
    if 'Unreported time' in df_filtered.columns and 'Machine number' in df_filtered.columns:
        unreported_machine = df_filtered.groupby('Machine number')['Unreported time'].sum().sort_values(
            ascending=False).head(10)
        unreported_machine = unreported_machine[unreported_machine > 0]  # Only show if time > 0
        if not unreported_machine.empty:
            fig_unreported = px.bar(unreported_machine,
                                    x=unreported_machine.index, y=unreported_machine.values,
                                    title='Machines with Highest Unreported Time',
                                    labels={'index': 'Machine Number', 'y': 'Unreported Time (minutes)'},
                                    color=unreported_machine.values, color_continuous_scale='Reds')
            st.plotly_chart(fig_unreported, use_container_width=True)
        else:
            st.info("No unreported time data to display.")
    else:
        st.info("Unreported time or Machine number column not available.")

# Quality Issues
st.markdown("### 🎯 Quality Issues Analysis")
col1_qual, col2_qual = st.columns(2)

with col1_qual:
    st.markdown("#### Rejection Rate by Item")
    if all(c in df_filtered.columns for c in ['Item code', 'Mfg qty', 'Rejected qty']):
        rejection_by_item = df_filtered.groupby('Item code').agg(
            Total_Mfg_Qty=('Mfg qty', 'sum'),
            Total_Rejected_Qty=('Rejected qty', 'sum')
        ).reset_index()
        rejection_by_item['Rejection Rate (%)'] = np.where(
            rejection_by_item['Total_Mfg_Qty'] > 0,
            (rejection_by_item['Total_Rejected_Qty'] / rejection_by_item['Total_Mfg_Qty']) * 100,
            0
        )
        rejection_by_item = rejection_by_item[rejection_by_item['Rejection Rate (%)'] > 0].sort_values(
            'Rejection Rate (%)', ascending=False).head(10)

        if not rejection_by_item.empty:
            fig_rej_item = px.bar(rejection_by_item,
                                  x='Item code', y='Rejection Rate (%)',
                                  title='Top Items by Rejection Rate',
                                  labels={'Item code': 'Item Code', 'Rejection Rate (%)': 'Rejection Rate (%)'},
                                  color='Rejection Rate (%)', color_continuous_scale='OrRd')
            st.plotly_chart(fig_rej_item, use_container_width=True)
        else:
            st.info("No items with rejections to display.")
    else:
        st.info("Required columns for rejection analysis by item are missing.")

with col2_qual:
    st.markdown("#### Production Performance by Operator")
    if all(c in df_filtered.columns for c in ['Operator name', 'Approved qty', 'Running time', 'Rejected qty']):
        operator_perf = df_filtered.groupby('Operator name').agg(
            Total_Approved_Qty=('Approved qty', 'sum'),
            Total_Rejected_Qty=('Rejected qty', 'sum'),
            Total_Running_Time_Mins=('Running time', 'sum')  # Assuming Running time is in minutes
        ).reset_index()
        operator_perf['Productivity (Approved Qty/Hr)'] = np.where(
            operator_perf['Total_Running_Time_Mins'] > 0,
            operator_perf['Total_Approved_Qty'] / (operator_perf['Total_Running_Time_Mins'] / 60),
            0
        )
        operator_perf['Rejection Rate (%)'] = np.where(
            (operator_perf['Total_Approved_Qty'] + operator_perf['Total_Rejected_Qty']) > 0,
            (operator_perf['Total_Rejected_Qty'] / (
                        operator_perf['Total_Approved_Qty'] + operator_perf['Total_Rejected_Qty'])) * 100,
            0
        )
        operator_perf_sorted = operator_perf[operator_perf['Productivity (Approved Qty/Hr)'] > 0].sort_values(
            'Productivity (Approved Qty/Hr)', ascending=False).head(10)

        if not operator_perf_sorted.empty:
            fig_op_perf = px.bar(operator_perf_sorted,
                                 x='Operator name', y='Productivity (Approved Qty/Hr)',
                                 title='Top Operators by Productivity (Approved Qty/Hour)',
                                 labels={'Operator name': 'Operator',
                                         'Productivity (Approved Qty/Hr)': 'Approved Parts per Hour'},
                                 color='Productivity (Approved Qty/Hr)', color_continuous_scale='Greens')
            st.plotly_chart(fig_op_perf, use_container_width=True)
        else:
            st.info("No operator performance data to display (check running times and approved quantities).")
    else:
        st.info("Required columns for operator performance analysis are missing.")

# Trend Analysis
st.markdown("## 📈 Trend Analysis")
if 'Date' in df_filtered.columns and len(df_filtered['Date'].dropna()) > 1:
    trend_metrics = ['Breakdown duration (in minutes)', 'Unreported time', 'Rejection Rate', 'Yield Rate',
                     'Machine Utilization', 'Mfg qty']
    # Filter out metrics not present in df_filtered
    available_trend_metrics = [m for m in trend_metrics if m in df_filtered.columns]

    if available_trend_metrics:
        trend_fig = create_trend_analysis(
            df_filtered,
            'Date',
            available_trend_metrics,
            'Key Metrics Over Time'
        )
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.info("Insufficient data for trend analysis after processing.")
    else:
        st.info("No suitable metrics found for trend analysis.")
else:
    st.info("Not enough date points or 'Date' column missing for trend analysis.")

# Machine Performance Heatmap
st.markdown("### 🔥 Machine Performance Heatmap (Utilization %)")
if all(c in df_filtered.columns for c in ['Machine number', 'Date', 'Machine Utilization']):
    heatmap_data = df_filtered[df_filtered['Date'].notna() & df_filtered['Machine number'].notna()]
    if not heatmap_data.empty:
        machine_daily_util = heatmap_data.groupby([heatmap_data['Date'].dt.date, 'Machine number'])[
            'Machine Utilization'].mean().reset_index()
        if not machine_daily_util.empty:
            pivot_util = machine_daily_util.pivot(index='Machine number', columns='Date', values='Machine Utilization')
            if not pivot_util.empty:
                # Sort columns (dates) and rows (machine numbers if numeric-like)
                pivot_util = pivot_util.reindex(sorted(pivot_util.columns), axis=1)
                try:  # Try to sort machine numbers numerically if they are like 'M1', 'M10'
                    pivot_util = pivot_util.reindex(sorted(pivot_util.index, key=lambda x: int(
                        str(x).replace('Machine', '').replace('M', '').strip()) if str(x).replace('Machine',
                                                                                                  '').replace('M',
                                                                                                              '').strip().isdigit() else float(
                        'inf')))
                except:
                    pivot_util = pivot_util.sort_index()  # Fallback to alphabetical sort

                fig_heatmap = px.imshow(pivot_util,
                                        title='Machine Utilization Heatmap (%)',
                                        color_continuous_scale='RdYlGn',  # Red-Yellow-Green
                                        aspect='auto',
                                        labels=dict(x="Date", y="Machine Number", color="Utilization %"))
                fig_heatmap.update_xaxes(type='category')  # Treat dates as categories on x-axis for discrete blocks
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Not enough data to create a pivot table for the machine utilization heatmap.")
        else:
            st.info("No daily machine utilization data aggregated for the heatmap.")
    else:
        st.info("No valid data for machine utilization heatmap after filtering.")
else:
    st.info("Machine number, Date, or Machine Utilization column missing for heatmap.")

# Action Items Table
st.markdown("## 🎯 Priority Action Items")
action_items = []

# Critical breakdown reasons
if 'Breakdown (entry)' in df_filtered.columns and 'Breakdown duration (in minutes)' in df_filtered.columns:
    top_breakdowns = df_filtered[df_filtered['Breakdown (entry)'] != 'Not Specified'].groupby('Breakdown (entry)')[
        'Breakdown duration (in minutes)'].sum().sort_values(ascending=False).head(3)
    for reason, duration in top_breakdowns.items():
        if duration > 0:
            action_items.append({
                'Priority': '🔴 High', 'Category': 'Breakdown',
                'Issue': f"Top Breakdown: {reason}",
                'Impact': f"{duration:.0f} minutes total lost",
                'Recommended Action': 'Investigate root cause, implement corrective/preventive actions.'
            })

# High rejection items (using previously calculated rejection_by_item if available)
if 'rejection_by_item' in locals() and not rejection_by_item.empty:
    for index, row_item in rejection_by_item.head(3).iterrows():  # Iterate over top 3 from the chart data
        if row_item['Rejection Rate (%)'] > 10:  # Example threshold
            action_items.append({
                'Priority': '🟡 Medium', 'Category': 'Quality',
                'Issue': f"High Rejection: Item {row_item['Item code']}",
                'Impact': f"{row_item['Rejection Rate (%)']:.1f}% rejection rate",
                'Recommended Action': 'Review process parameters, material quality, and operator training for this item.'
            })

# High unreported time machines (using previously calculated unreported_machine if available)
if 'unreported_machine' in locals() and not unreported_machine.empty:
    for machine, time_lost in unreported_machine.head(3).items():  # Iterate over top 3
        if time_lost > 60:  # Example threshold: more than 1 hour
            action_items.append({
                'Priority': '🟠 Medium', 'Category': 'Efficiency',
                'Issue': f"High Unreported Time: Machine {machine}",
                'Impact': f"{time_lost:.0f} minutes unreported",
                'Recommended Action': 'Investigate reasons for unreported time, improve time logging practices.'
            })

# Missing Breakdown Form Entries
if 'Missing_Form_Entry' in df_filtered.columns and df_filtered['Missing_Form_Entry'].sum() > 0:
    missing_count = df_filtered['Missing_Form_Entry'].sum()
    action_items.append({
        'Priority': '🔵 Low', 'Category': 'Data Integrity',
        'Issue': f"{missing_count} PRD records flagged breakdown but no form entry",
        'Impact': 'Potential underreporting of breakdown details.',
        'Recommended Action': 'Review PRD records vs. Form submissions. Ensure operators complete forms for all breakdowns.'
    })

if action_items:
    action_df = pd.DataFrame(action_items)
    st.dataframe(action_df, use_container_width=True, hide_index=True)
else:
    st.success("✅ No critical action items automatically identified based on current thresholds.")

# Data Export
st.markdown("## 📥 Data Export")
col1_exp, col2_exp = st.columns(2)

with col1_exp:
    # Provide df_filtered for download as it reflects user's selections
    csv_filtered = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Filtered Data (CSV)",
        data=csv_filtered,
        file_name=f"filtered_production_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

with col2_exp:
    # Provide df_processed (pre-filter) for download
    csv_processed = df_processed.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📋 Download All Processed Data (CSV)",
        data=csv_processed,
        file_name=f"all_processed_production_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

# Footer
st.markdown("---")
st.markdown(
    f"*Dashboard reflects data up to {df_processed['Date'].max().strftime('%Y-%m-%d') if not df_processed['Date'].dropna().empty else 'N/A'}. Last refreshed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")