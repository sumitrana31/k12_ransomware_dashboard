import requests
import re
import pandas as pd
import logging
from datetime import datetime, timedelta
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import os

# Set up logging
logging.basicConfig(
    filename='ransomware_k12.log',
    level=logging.INFO,  # Change to DEBUG for more verbosity
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Keywords to identify K-12 schools
school_keywords = [
    'school', 'school district', 'elementary', 'middle school',
    'high school', 'public schools', 'unified school district',
    'independent school district', 'charter school',
    'primary school', 'secondary school', 'k12', 'k-12',
    'preparatory school', 'prep school', 'public education',
    'private school', 'learning center', 'montessori',
    'kindergarten', 'nursery school', 'daycare', 'head start',
    'grade school', 'grammar school', 'junior high', 'senior high',
    'district schools', 'public school system'
]

# Exclusion keywords to filter out non-K12 educational institutions
exclusion_keywords = [
    'university', 'college', 'institute', 'academy', 'training center',
    'technical school', 'online education', 'professional development',
    'tappi', 'pulp', 'paper industry', 'enrollment services'
]

# Compile the regex pattern once with word boundaries
pattern = re.compile(r'\b(' + '|'.join(school_keywords) + r')\b', re.IGNORECASE)

# Global variables for caching
TOTAL_DATA_FILE = 'total_data.csv'
K12_DATA_FILE = 'k12_data.csv'
CACHE_EXPIRY = timedelta(hours=1)  # Data refresh interval

# Function to fetch ransomware data from the API and save to CSV
def fetch_and_save_ransomware_data():
    years = [2021, 2022, 2023, 2024]  # Adjust years as needed
    data = []
    for year in years:
        url = f"https://api.ransomware.live/victims/{year}"
        headers = {"accept": "application/json"}
        logging.info(f"Fetching data for the year {year} from Ransomware.live API.")
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            year_data = response.json()
            data.extend(year_data)
            logging.info(f"Data for year {year} fetched successfully.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data for year {year}: {e}")

    # Prepare DataFrame and save to CSV
    data_rows = []
    for incident in data:
        victim_name = incident.get('victim_name')
        if not victim_name:
            victim_name = incident.get('post_title') or incident.get('title') or 'N/A'

        data_row = {
            'Victim Name': victim_name,
            'Group Name': incident.get('group_name', 'Unknown'),
            'Date Published': incident.get('published', 'Unknown'),
            'Country': incident.get('country', 'Unknown'),
            'Description': incident.get('description', ''),
            'Website': incident.get('website', ''),
            'Activity': incident.get('activity', ''),
            'Post URL': incident.get('post_url', ''),
            'Discovered': incident.get('discovered', ''),
        }
        data_rows.append(data_row)

    df_total = pd.DataFrame(data_rows)
    # Process date fields
    for date_field in ['Date Published', 'Discovered']:
        df_total[date_field] = pd.to_datetime(df_total[date_field], errors='coerce')
    df_total['Country'] = df_total['Country'].str.upper()

    # Save total data to CSV
    df_total.to_csv(TOTAL_DATA_FILE, index=False)
    logging.info(f"Total data saved to {TOTAL_DATA_FILE}")

    # Filter for K-12 schools and save to CSV
    df_total['is_k12'] = df_total.apply(is_us_k12_school, axis=1)
    df_k12 = df_total[df_total['is_k12']].copy()
    df_k12.to_csv(K12_DATA_FILE, index=False)
    logging.info(f"K-12 school data saved to {K12_DATA_FILE}")

# Function to check if an incident involves a US K-12 school
def is_us_k12_school(row):
    country = row.get('Country', '').upper()
    victim_name = row.get('Victim Name', '')
    description = row.get('Description', '')
    combined_text = ' '.join([victim_name, description]).lower()

    if country != 'US':
        return False

    if any(ex_keyword in combined_text for ex_keyword in exclusion_keywords):
        return False

    if pattern.search(combined_text):
        return True
    else:
        return False

# Function to load data from CSV files
def load_data():
    # Check if data files exist and are fresh
    if os.path.exists(TOTAL_DATA_FILE) and os.path.exists(K12_DATA_FILE):
        total_data_mtime = datetime.fromtimestamp(os.path.getmtime(TOTAL_DATA_FILE))
        k12_data_mtime = datetime.fromtimestamp(os.path.getmtime(K12_DATA_FILE))
        now = datetime.now()
        if now - total_data_mtime < CACHE_EXPIRY and now - k12_data_mtime < CACHE_EXPIRY:
            # Load data from CSV files
            df_total = pd.read_csv(TOTAL_DATA_FILE, parse_dates=['Date Published', 'Discovered'])
            df_k12 = pd.read_csv(K12_DATA_FILE, parse_dates=['Date Published', 'Discovered'])
            df_total['Country'] = df_total['Country'].str.upper()
            df_k12['Country'] = df_k12['Country'].str.upper()
            return df_total, df_k12

    # If files don't exist or are outdated, fetch new data
    fetch_and_save_ransomware_data()
    df_total = pd.read_csv(TOTAL_DATA_FILE, parse_dates=['Date Published', 'Discovered'])
    df_k12 = pd.read_csv(K12_DATA_FILE, parse_dates=['Date Published', 'Discovered'])
    df_total['Country'] = df_total['Country'].str.upper()
    df_k12['Country'] = df_k12['Country'].str.upper()
    return df_total, df_k12

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = 'US K-12 Ransomware Incidents Dashboard'
server = app.server  # For deployment if needed

# Load data
df_total, df_k12 = load_data()

# Compute summary statistics
total_attacks = len(df_total)
us_attacks = len(df_total[df_total['Country'] == 'US'])
k12_attacks = len(df_k12)

# Prepare df_k12
df_k12['Year-Month'] = df_k12['Date Published'].dt.to_period('M').astype(str)

# Get latest 5 school attacks
latest_attacks = df_k12.sort_values(by='Date Published', ascending=False).head(5)
latest_attacks_list = [
    dbc.ListGroupItem([
        html.H5(latest_attacks.iloc[i]['Victim Name']),
        html.P(f"Date Published: {latest_attacks.iloc[i]['Date Published'].date()}"),
        html.P(f"Group Name: {latest_attacks.iloc[i]['Group Name']}"),
    ]) for i in range(len(latest_attacks))
]

# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1('US K-12 Ransomware Incidents Dashboard'), width=12)
    ], className='my-3'),

    # Summary Statistics
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Total Attacks', className='card-title'),
                html.H2(id='total-attacks', className='card-text'),
            ])
        ], color='primary', inverse=True), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Total US Attacks', className='card-title'),
                html.H2(id='us-attacks', className='card-text'),
            ])
        ], color='success', inverse=True), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Total Attacks on US K-12 Schools', className='card-title'),
                html.H2(id='k12-attacks', className='card-text'),
            ])
        ], color='danger', inverse=True), width=4),
    ], className='my-4'),

    # Latest 5 Schools Attacked
    dbc.Row([
        dbc.Col(html.H3('Latest 5 Schools Attacked'), width=12)
    ]),
    dbc.Row([
        dbc.Col(
            dbc.ListGroup(
                id='latest-attacks-list',
                children=latest_attacks_list
            ), width=12
        )
    ], className='my-4'),

    # Filters
    dbc.Row([
        dbc.Col([
            html.Label('Date Range:'),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=df_k12['Date Published'].min().date(),
                max_date_allowed=df_k12['Date Published'].max().date(),
                start_date=df_k12['Date Published'].min().date(),
                end_date=df_k12['Date Published'].max().date()
            ),
        ], width=6),
        dbc.Col([
            html.Label('Ransomware Group:'),
            dcc.Dropdown(
                id='group-dropdown',
                options=[{'label': grp, 'value': grp} for grp in sorted(df_k12['Group Name'].unique())],
                value=df_k12['Group Name'].unique().tolist(),
                multi=True
            ),
        ], width=6),
    ], className='my-3'),

    # Graphs
    dbc.Row([
        dbc.Col(dcc.Graph(id='time-series-chart'), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='group-bar-chart'), width=12)
    ]),

    # Data Table
    dbc.Row([
        dbc.Col(html.H2('Detailed Incident Data'), width=12)
    ], className='my-3'),
    dbc.Row([
        dbc.Col(
            dash_table.DataTable(
                id='incident-table',
                columns=[{'name': col, 'id': col, 'type': 'text'} for col in df_k12.columns if col != 'is_k12'],
                data=df_k12.sort_values(by='Date Published', ascending=False).to_dict('records'),
                style_cell={
                    'textAlign': 'left',  # Align text to the left
                    'padding': '5px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_table={
                    'overflowX': 'auto',
                    'maxHeight': '800px',
                    'overflowY': 'scroll',
                },
                fixed_rows={'headers': True},
                page_action='none',  # Show all data without pagination
                sort_action='native',
                filter_action='native',
            ), width=12
        )
    ]),

    # Interval Component for Data Refresh
    dcc.Interval(
        id='interval-component',
        interval=60*60*1000,  # Refresh every hour (milliseconds)
        n_intervals=0
    )
], fluid=True)

# Callbacks for interactivity and data refresh
@app.callback(
    [Output('time-series-chart', 'figure'),
     Output('group-bar-chart', 'figure'),
     Output('incident-table', 'data'),
     Output('total-attacks', 'children'),
     Output('us-attacks', 'children'),
     Output('k12-attacks', 'children'),
     Output('latest-attacks-list', 'children')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('group-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(start_date, end_date, selected_groups, n_intervals):
    # Load data (will refresh if cache expired)
    df_total, df_k12 = load_data()

    # Update summary statistics
    total_attacks = len(df_total)
    us_attacks = len(df_total[df_total['Country'] == 'US'])
    k12_attacks = len(df_k12)

    # Update latest attacks
    latest_attacks = df_k12.sort_values(by='Date Published', ascending=False).head(5)
    latest_attacks_list = [
        dbc.ListGroupItem([
            html.H5(latest_attacks.iloc[i]['Victim Name']),
            html.P(f"Date Published: {latest_attacks.iloc[i]['Date Published'].date()}"),
            html.P(f"Group Name: {latest_attacks.iloc[i]['Group Name']}"),
        ]) for i in range(len(latest_attacks))
    ]

    # Prepare df_k12
    df_k12['Year-Month'] = df_k12['Date Published'].dt.to_period('M').astype(str)

    # Update filters
    mask = (
        (df_k12['Date Published'] >= pd.to_datetime(start_date)) &
        (df_k12['Date Published'] <= pd.to_datetime(end_date)) &
        (df_k12['Group Name'].isin(selected_groups))
    )
    filtered_df = df_k12.loc[mask]

    # Time Series Chart
    incidents_over_time = filtered_df.groupby('Year-Month').size().reset_index(name='Incidents')
    fig_time_series = px.line(
        incidents_over_time,
        x='Year-Month',
        y='Incidents',
        title='Incidents Over Time',
        markers=True
    )

    # Bar Chart of Ransomware Groups
    group_counts = filtered_df['Group Name'].value_counts().reset_index()
    group_counts.columns = ['Group Name', 'Incidents']
    fig_group_bar = px.bar(
        group_counts,
        x='Group Name',
        y='Incidents',
        title='Incidents per Ransomware Group'
    )

    # Data Table
    table_data = filtered_df.sort_values(by='Date Published', ascending=False).to_dict('records')

    # Return updated components
    return (
        fig_time_series,
        fig_group_bar,
        table_data,
        str(total_attacks),
        str(us_attacks),
        str(k12_attacks),
        latest_attacks_list
    )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
