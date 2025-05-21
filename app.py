import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import datetime
import os

# --- Configuration ---
# Replace with your actual Google Sheet URL
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/18ZbDcL0e053w1-IbxX-OflUOs1z07QTcQYUHtLGwl1k/edit?gid=0#gid=0"
# Replace with the name of the specific sheet/tab you want to use
WORKSHEET_NAME = "PRD"  # e.g., "Production Log" or whatever your tab is named


# --- Google Sheets Connection ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour to reduce API calls
def get_google_sheet_data():
    client = None  # Initialize client to None

    # Attempt to use Streamlit secrets first
    if "gcp_service_account" in st.secrets:
        try:
            creds_dict = st.secrets["gcp_service_account"]
            # Ensure creds_dict is a dictionary (Streamlit Cloud parses TOML to dict)
            if not isinstance(creds_dict, dict):
                st.warning("Streamlit secret 'gcp_service_account' is not a valid dictionary. Attempting local credentials.")
            else:
                client = gspread.service_account_from_dict(creds_dict)
                # st.info("Using Streamlit secrets for Google Sheets.") # Optional: for debugging
        except Exception as e:
            st.warning(f"Error initializing client from Streamlit secrets: {e}. Attempting local credentials.")
            client = None # Ensure client is None if secrets method fails
    else:
        st.info("Streamlit secret 'gcp_service_account' not found. Attempting local 'credentials.json'.")

    # If client is still None (secrets not found or failed), try local credentials.json
    if client is None:
        if not os.path.exists("credentials.json"):
            st.error("Local 'credentials.json' file not found. Cannot connect to Google Sheets.")
            return pd.DataFrame() # Return empty DataFrame as a fallback

        try:
            scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
            local_creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
            client = gspread.authorize(local_creds)
            # st.info("Using local 'credentials.json' for Google Sheets.") # Optional: for debugging
        except Exception as e:
            st.error(f"Error loading local 'credentials.json': {e}")
            return pd.DataFrame() # Return empty DataFrame

    # If client is still None at this point, something went wrong with both methods
    if client is None:
        st.error("Failed to initialize Google Sheets client using both Streamlit secrets and local credentials.")
        return pd.DataFrame()

    # Proceed to fetch data from the Google Sheet
    try:
        spreadsheet = client.open_by_url(GOOGLE_SHEET_URL)
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        records = worksheet.get_all_records()
        # If sheet is empty, get_all_records() returns an empty list,
        # pd.DataFrame([]) is an empty DataFrame, which is fine.
        return pd.DataFrame(records)
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Spreadsheet not found. Check the GOOGLE_SHEET_URL: {GOOGLE_SHEET_URL}")
        return pd.DataFrame()
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Worksheet '{WORKSHEET_NAME}' not found in the spreadsheet. Check WORKSHEET_NAME.")
        return pd.DataFrame()
    except gspread.exceptions.APIError as api_e:
        st.error(f"Google Sheets API Error: {api_e}. "
                 "This could be due to insufficient permissions for the service account on the sheet, "
                 "or incorrect API enabling in Google Cloud Console.")
        return pd.DataFrame()
    except Exception as sheet_error:
        st.error(f"An unexpected error occurred while accessing the Google Sheet: {sheet_error}")
        return pd.DataFrame()


# --- Data Loading and Preprocessing ---
@st.cache_data(ttl=3600)  # Cache the processed DataFrame too
def process_data(df):
    # If DataFrame is empty, return it as is
    if df.empty:
        return df

    # Convert relevant columns to appropriate data types
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Handling time columns:
    # If Machine start/end time are just 'HH:MM' or 'HH:MM:SS' strings, convert to datetime.time objects
    for col in ['Machine start time', 'Machine End time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce').dt.time

    # Numeric columns: Use errors='coerce' to turn non-numeric values into NaN
    numeric_cols = [
        'Running time', 'Process time (Machining)', 'Process time (Setup)',
        'Mfg qty', 'Rejected qty', 'Approved qty', 'Down time (duration)'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)  # Fill NaN with 0 for calculations

    # Calculate derived metrics
    df['Total Process Time'] = df['Process time (Machining)'] + df['Process time (Setup)']
    df['Yield Rate'] = (df['Approved qty'] / df['Mfg qty']) * 100
    df['Yield Rate'] = df['Yield Rate'].fillna(0).replace([float('inf'), -float('inf')], 0)  # Handle division by zero

    # Calculate Machine Utilization
    df['Productive Machine Time'] = df['Process time (Machining)'] + df['Process time (Setup)']

    # Calculate Utilization - THIS WAS THE ERROR - df_processed was used before definition
    df['Utilization'] = (df['Process time (Machining)'] + df['Process time (Setup)']) / (
            df['Running time'] + df['Down time (duration)'])
    df['Utilization'] = df['Utilization'].fillna(0).replace([float('inf'), -float('inf')],
                                                            0) * 100  # Convert to percentage

    return df


# --- Streamlit Dashboard Layout ---
st.set_page_config(layout="wide", page_title="Manufacturing Efficiency Dashboard")

st.title("🏭 Manufacturing Efficiency Dashboard")

# Load and process data
df = get_google_sheet_data()

# Check if data is empty before processing
if df.empty:
    st.error("Unable to load data from Google Sheets. Please check your connection and credentials.")
    st.stop()

# Process the data
df_processed = process_data(df.copy())  # Use a copy to avoid modifying the cached raw data

# Ensure 'Date' is correctly set as index for time series charts or for filtering
df_processed = df_processed.sort_values(by='Date').reset_index(drop=True)

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Date Range Filter
min_date = df_processed['Date'].min().date() if not df_processed[
    'Date'].isnull().all() else datetime.date.today() - datetime.timedelta(days=30)
max_date = df_processed['Date'].max().date() if not df_processed['Date'].isnull().all() else datetime.date.today()

date_selection = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_selection) == 2:
    start_date, end_date = date_selection
    df_filtered = df_processed[
        (df_processed['Date'].dt.date >= start_date) & (df_processed['Date'].dt.date <= end_date)]
else:
    df_filtered = df_processed  # Show all if date range is not complete

# Machine Filter
all_machines = ['All'] + sorted(df_filtered['Machine number'].unique().tolist())
selected_machine = st.sidebar.selectbox("Select Machine Number", all_machines)
if selected_machine != 'All':
    df_filtered = df_filtered[df_filtered['Machine number'] == selected_machine]

# Shift Filter
all_shifts = ['All'] + sorted(df_filtered['Shift'].unique().tolist())
selected_shift = st.sidebar.selectbox("Select Shift", all_shifts)
if selected_shift != 'All':
    df_filtered = df_filtered[df_filtered['Shift'] == selected_shift]

# Operator Filter
all_operators = ['All'] + sorted(df_filtered['Operator name'].unique().tolist())
selected_operator = st.sidebar.selectbox("Select Operator", all_operators)
if selected_operator != 'All':
    df_filtered = df_filtered[df_filtered['Operator name'] == selected_operator]

# --- Main Dashboard Content ---

# Check if filtered data is empty
if df_filtered.empty:
    st.warning("No data available for the selected filters.")
else:
    # --- KPIs ---
    st.header("Overall Performance KPIs")
    col1, col2, col3, col4, col5 = st.columns(5)

    total_mfg_qty = df_filtered['Mfg qty'].sum()
    total_approved_qty = df_filtered['Approved qty'].sum()
    total_rejected_qty = df_filtered['Rejected qty'].sum()
    total_downtime_hours = df_filtered['Down time (duration)'].sum()

    overall_yield_rate = (total_approved_qty / total_mfg_qty) * 100 if total_mfg_qty > 0 else 0
    overall_rejection_rate = (total_rejected_qty / total_mfg_qty) * 100 if total_mfg_qty > 0 else 0

    col1.metric("Total Mfg Qty", f"{int(total_mfg_qty):,}")
    col2.metric("Total Approved Qty", f"{int(total_approved_qty):,}")
    col3.metric("Total Rejected Qty", f"{int(total_rejected_qty):,}")
    col4.metric("Overall Yield Rate", f"{overall_yield_rate:.2f}%")
    col5.metric("Total Downtime (hours)", f"{total_downtime_hours:,.2f}")

    st.markdown("---")

    # --- Charts and Graphs ---
    st.header("Detailed Performance Analysis")

    # Row 1: Downtime Reasons and Machine Performance
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Downtime Reasons by Duration")
        if 'Down time (Reason)' in df_filtered.columns:
            downtime_by_reason = df_filtered.groupby('Down time (Reason)')['Down time (duration)'].sum().sort_values(
                ascending=False)
            if not downtime_by_reason.empty:
                st.bar_chart(downtime_by_reason)
            else:
                st.info("No downtime data for selected filters.")
        else:
            st.info("Downtime reason column not found in the data.")

    with chart_col2:
        st.subheader("Approved Quantity by Machine")
        approved_by_machine = df_filtered.groupby('Machine number')['Approved qty'].sum().sort_values(ascending=False)
        if not approved_by_machine.empty:
            st.bar_chart(approved_by_machine)
        else:
            st.info("No approved quantity data for selected filters.")

    # Row 2: Yield Rate Trend and Operator Performance
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader("Daily Yield Rate Trend")
        # Ensure 'Date' column is not null for grouping
        daily_yield = df_filtered.dropna(subset=['Date']).groupby(df_filtered['Date'].dt.date).agg(
            total_mfg=('Mfg qty', 'sum'),
            total_approved=('Approved qty', 'sum')
        ).reset_index()

        if not daily_yield.empty:
            daily_yield['Daily Yield Rate'] = (daily_yield['total_approved'] / daily_yield['total_mfg']) * 100
            daily_yield['Daily Yield Rate'] = daily_yield['Daily Yield Rate'].fillna(0).replace(
                [float('inf'), -float('inf')], 0)
            st.line_chart(daily_yield.set_index('Date')['Daily Yield Rate'])
        else:
            st.info("No daily yield data for selected filters.")

    with chart_col4:
        st.subheader("Approved Quantity by Operator")
        approved_by_operator = df_filtered.groupby('Operator name')['Approved qty'].sum().sort_values(ascending=False)
        if not approved_by_operator.empty:
            st.bar_chart(approved_by_operator)
        else:
            st.info("No approved quantity data for selected filters.")

    # Row 3: Setup Time Analysis
    st.subheader("Setup Time Analysis by Item Code / Operation")
    if all(col in df_filtered.columns for col in ['Item code', 'Operation or Process description']):
        setup_time_summary = df_filtered.groupby(['Item code', 'Operation or Process description'])[
            'Process time (Setup)'].sum().reset_index()
        if not setup_time_summary.empty:
            st.dataframe(setup_time_summary.sort_values(by='Process time (Setup)', ascending=False))
        else:
            st.info("No setup time data for selected filters.")
    else:
        st.info("Required columns for setup time analysis are missing in the data.")

    st.markdown("---")
    st.header("Raw Data Preview")
    st.dataframe(df_filtered.head(100))  # Show first 100 rows of filtered data