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
import plotly.graph_objects as go
import os

# Set up logging
logging.basicConfig(
    filename='ransomware_k12.log',
    level=logging.INFO,
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

def fetch_and_save_ransomware_data():
    years = [2021, 2022, 2023, 2024]
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
    for date_field in ['Date Published', 'Discovered']:
        df_total[date_field] = pd.to_datetime(df_total[date_field], errors='coerce')
    df_total['Country'] = df_total['Country'].str.upper()

    df_total.to_csv(TOTAL_DATA_FILE, index=False)
    logging.info(f"Total data saved to {TOTAL_DATA_FILE}")

    df_total['is_k12'] = df_total.apply(is_us_k12_school, axis=1)
    df_k12 = df_total[df_total['is_k12']].copy()
    df_k12.to_csv(K12_DATA_FILE, index=False)
    logging.info(f"K-12 school data saved to {K12_DATA_FILE}")

def is_us_k12_school(row):
    country = row.get('Country', '').upper()
    victim_name = row.get('Victim Name', '')
    description = row.get('Description', '')
    combined_text = ' '.join([victim_name, description]).lower()

    if country != 'US':
        return False

    if any(ex_keyword in combined_text for ex_keyword in exclusion_keywords):
        return False

    return bool(pattern.search(combined_text))

def load_data():
    if os.path.exists(TOTAL_DATA_FILE) and os.path.exists(K12_DATA_FILE):
        total_data_mtime = datetime.fromtimestamp(os.path.getmtime(TOTAL_DATA_FILE))
        k12_data_mtime = datetime.fromtimestamp(os.path.getmtime(K12_DATA_FILE))
        now = datetime.now()
        if now - total_data_mtime < CACHE_EXPIRY and now - k12_data_mtime < CACHE_EXPIRY:
            df_total = pd.read_csv(TOTAL_DATA_FILE, parse_dates=['Date Published', 'Discovered'])
            df_k12 = pd.read_csv(K12_DATA_FILE, parse_dates=['Date Published', 'Discovered'])
            df_total['Country'] = df_total['Country'].str.upper()
            df_k12['Country'] = df_k12['Country'].str.upper()
            return df_total, df_k12

    fetch_and_save_ransomware_data()
    df_total = pd.read_csv(TOTAL_DATA_FILE, parse_dates=['Date Published', 'Discovered'])
    df_k12 = pd.read_csv(K12_DATA_FILE, parse_dates=['Date Published', 'Discovered'])
    df_total['Country'] = df_total['Country'].str.upper()
    df_k12['Country'] = df_k12['Country'].str.upper()
    return df_total, df_k12

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = 'US K-12 Ransomware Incidents Dashboard'
server = app.server

# Load initial data
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

# Layout with improved styling
app.layout = dbc.Container([
    # Header with improved styling
    dbc.Row([
        dbc.Col(html.H1('US K-12 Ransomware Incidents Dashboard', 
                       className='text-center text-primary mb-4'), 
                width=12)
    ], className='mt-4'),

    # Summary Statistics with enhanced cards
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Total Attacks', className='card-title text-center'),
                html.H2(id='total-attacks', className='card-text text-center'),
            ])
        ], color='primary', inverse=True, className='shadow'), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Total US Attacks', className='card-title text-center'),
                html.H2(id='us-attacks', className='card-text text-center'),
            ])
        ], color='success', inverse=True, className='shadow'), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Total Attacks on US K-12 Schools', className='card-title text-center'),
                html.H2(id='k12-attacks', className='card-text text-center'),
            ])
        ], color='danger', inverse=True, className='shadow'), width=4),
    ], className='mb-4'),

    # Latest Attacks section with improved styling
    dbc.Row([
        dbc.Col(html.H3('Latest School Attacks', className='text-primary mb-3'), width=12)
    ]),
    dbc.Row([
        dbc.Col(
            dbc.ListGroup(
                id='latest-attacks-list',
                children=latest_attacks_list,
                className='shadow-sm'
            ), width=12
        )
    ], className='mb-4'),

    # Filters with improved styling
    dbc.Row([
        dbc.Col([
            html.Label('Date Range:', className='font-weight-bold'),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=df_k12['Date Published'].min().date(),
                max_date_allowed=df_k12['Date Published'].max().date(),
                start_date=df_k12['Date Published'].min().date(),
                end_date=df_k12['Date Published'].max().date(),
                className='mb-3'
            ),
        ], width=6),
        dbc.Col([
            html.Label('Ransomware Group:', className='font-weight-bold'),
            dcc.Dropdown(
                id='group-dropdown',
                options=[{'label': grp, 'value': grp} for grp in sorted(df_k12['Group Name'].unique())],
                value=df_k12['Group Name'].unique().tolist(),
                multi=True,
                className='mb-3'
            ),
        ], width=6),
    ], className='mb-4'),

    # Graphs with improved styling
    dbc.Row([
        dbc.Col(dcc.Graph(id='time-series-chart', className='shadow-sm'), width=12)
    ], className='mb-4'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='group-bar-chart', className='shadow-sm'), width=12)
    ], className='mb-4'),

    # Data Table with improved styling
    dbc.Row([
        dbc.Col(html.H2('Detailed Incident Data', className='text-primary mb-3'), width=12)
    ]),
    dbc.Row([
        dbc.Col(
            dash_table.DataTable(
                id='incident-table',
                columns=[{'name': col, 'id': col, 'type': 'text'} 
                        for col in df_k12.columns if col != 'is_k12'],
                data=df_k12.sort_values(by='Date Published', ascending=False).to_dict('records'),
                style_cell={
                    'textAlign': 'left',
                    'padding': '12px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'fontSize': '14px',
                    'fontFamily': 'Arial, sans-serif',
                    'minWidth': '150px'
                },
                style_header={
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': 'bold',
                    'border': '1px solid #dee2e6',
                    'color': '#2c3e50'
                },
                style_data={
                    'backgroundColor': 'white',
                    'border': '1px solid #dee2e6',
                    'color': '#2c3e50'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f8f9fa'
                    }
                ],
                style_table={
                    'overflowX': 'auto',
                    'maxHeight': '800px',
                    'overflowY': 'scroll',
                    'border': '1px solid #dee2e6',
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)'
                },
                fixed_rows={'headers': True},
                page_action='none',
                sort_action='native',
                filter_action='native'
            ), width=12
        )
    ], className='mb-4'),

    # Interval Component for Data Refresh
    dcc.Interval(
        id='interval-component',
        interval=60*60*1000,  # Refresh every hour
        n_intervals=0
    )
], fluid=True, className='px-4 py-3')

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

    # Enhanced Time Series Chart with sophisticated styling
    incidents_over_time = filtered_df.groupby('Year-Month').size().reset_index(name='Incidents')
    
    # Calculate moving average and year-over-year comparison
    incidents_over_time['Moving_Average'] = incidents_over_time['Incidents'].rolling(window=3, center=True).mean()
    
    fig_time_series = go.Figure()

    # Add sophisticated area fill
    fig_time_series.add_trace(
        go.Scatter(
            x=incidents_over_time['Year-Month'],
            y=incidents_over_time['Incidents'],
            fill='tozeroy',
            fillcolor='rgba(45, 152, 218, 0.1)',
            mode='none',
            name='Area',
            showlegend=False
        )
    )

    # Add gradient-styled area for emphasis
    fig_time_series.add_trace(
        go.Scatter(
            x=incidents_over_time['Year-Month'],
            y=incidents_over_time['Incidents'],
            fill='tonexty',
            fillcolor='rgba(45, 152, 218, 0.05)',
            mode='none',
            name='Gradient',
            showlegend=False
        )
    )

    # Add main line with custom styling
    fig_time_series.add_trace(
        go.Scatter(
            x=incidents_over_time['Year-Month'],
            y=incidents_over_time['Incidents'],
            mode='lines+markers',
            name='Monthly Incidents',
            line=dict(
                color='#2980b9',
                width=3,
                shape='spline'  # Smooth curve
            ),
            marker=dict(
                size=10,
                symbol='circle',
                color='white',
                line=dict(
                    color='#2980b9',
                    width=2
                )
            )
        )
    )

    # Add moving average with sophisticated styling
    fig_time_series.add_trace(
        go.Scatter(
            x=incidents_over_time['Year-Month'],
            y=incidents_over_time['Moving_Average'],
            mode='lines',
            name='3-Month Trend',
            line=dict(
                color='#e74c3c',
                width=2,
                dash='dot'
            )
        )
    )

    # Add min/max reference lines
    min_incidents = incidents_over_time['Incidents'].min()
    max_incidents = incidents_over_time['Incidents'].max()
    avg_incidents = incidents_over_time['Incidents'].mean()

    # Update layout with sophisticated styling
    fig_time_series.update_layout(
        title=dict(
            text='K-12 Ransomware Incidents Over Time',
            x=0.5,
            y=0.95,
            xanchor='center',
            font=dict(
                size=24,
                color='#2c3e50',
                family='Arial Black'
            )
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            title=dict(
                text='Month',
                font=dict(size=14, family='Arial')
            ),
            tickangle=45,
            showgrid=True,
            gridcolor='rgba(189,189,189,0.2)',
            showline=True,
            linecolor='#2c3e50',
            linewidth=2,
            ticks='outside',
            tickfont=dict(size=12, family='Arial'),
            mirror=True
        ),
        yaxis=dict(
            title=dict(
                text='Number of Incidents',
                font=dict(size=14, family='Arial')
            ),
            showgrid=True,
            gridcolor='rgba(189,189,189,0.2)',
            showline=True,
            linecolor='#2c3e50',
            linewidth=2,
            ticks='outside',
            tickfont=dict(size=12, family='Arial'),
            mirror=True,
            rangemode='tozero',
            zeroline=True,
            zerolinecolor='#2c3e50',
            zerolinewidth=1
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#2c3e50',
            borderwidth=1,
            font=dict(size=12, family='Arial')
        ),
        margin=dict(t=100, l=80, r=40, b=80),
        shapes=[
            # Add horizontal reference lines
            dict(
                type='line',
                x0=0,
                x1=1,
                y0=avg_incidents,
                y1=avg_incidents,
                yref='y',
                xref='paper',
                line=dict(
                    color='rgba(44, 62, 80, 0.3)',
                    width=1,
                    dash='dash'
                )
            )
        ],
        annotations=[
            # Add latest value annotation
            dict(
                x=incidents_over_time['Year-Month'].iloc[-1],
                y=incidents_over_time['Incidents'].iloc[-1],
                text=f"Latest: {incidents_over_time['Incidents'].iloc[-1]}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#2c3e50',
                ax=40,
                ay=-40,
                font=dict(size=12, color='#2c3e50', family='Arial'),
                bgcolor='white',
                bordercolor='#2c3e50',
                borderwidth=1,
                borderpad=4,
                opacity=0.9
            ),
            # Add average value annotation
            dict(
                x=incidents_over_time['Year-Month'].iloc[0],
                y=avg_incidents,
                text=f"Average: {avg_incidents:.1f}",
                showarrow=False,
                yshift=10,
                font=dict(size=10, color='#7f8c8d', family='Arial'),
                xshift=-10
            )
        ]
    )

    # Add hover template
    fig_time_series.update_traces(
        hovertemplate="<b>Date</b>: %{x}<br>" +
                     "<b>Incidents</b>: %{y}<br>" +
                     "<extra></extra>",
    )

    # Add subtle grid pattern
    for y in range(int(min_incidents), int(max_incidents) + 2):
        fig_time_series.add_shape(
            type='line',
            x0=0,
            x1=1,
            y0=y,
            y1=y,
            xref='paper',
            yref='y',
            line=dict(
                color='rgba(189,189,189,0.2)',
                width=1
            )
        )

    # Add hover template
    fig_time_series.update_traces(
        hovertemplate="<b>Month</b>: %{x}<br>" +
                     "<b>Incidents</b>: %{y}<br>" +
                     "<extra></extra>"  # This removes the secondary box
    )

    # Bar Chart of Ransomware Groups with enhanced styling
    group_counts = filtered_df['Group Name'].value_counts().reset_index()
    group_counts.columns = ['Group Name', 'Incidents']
    
    fig_group_bar = go.Figure()

    # Add main bars
    fig_group_bar.add_trace(
        go.Bar(
            x=group_counts['Group Name'],
            y=group_counts['Incidents'],
            marker_color='#2ecc71',
            marker_line_color='#27ae60',
            marker_line_width=1.5,
            opacity=0.9,
            name='Incidents'
        )
    )

    # Update layout with sophisticated styling
    fig_group_bar.update_layout(
        title=dict(
            text='Incidents per Ransomware Group',
            x=0.5,
            y=0.95,
            xanchor='center',
            font=dict(
                size=24,
                color='#2c3e50',
                family='Arial Black'
            )
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            title=dict(
                text='Group Name',
                font=dict(size=14, family='Arial')
            ),
            tickangle=45,
            showgrid=True,
            gridcolor='rgba(189,189,189,0.2)',
            showline=True,
            linecolor='#2c3e50',
            linewidth=2,
            ticks='outside',
            tickfont=dict(size=12, family='Arial'),
            mirror=True
        ),
        yaxis=dict(
            title=dict(
                text='Number of Incidents',
                font=dict(size=14, family='Arial')
            ),
            showgrid=True,
            gridcolor='rgba(189,189,189,0.2)',
            showline=True,
            linecolor='#2c3e50',
            linewidth=2,
            ticks='outside',
            tickfont=dict(size=12, family='Arial'),
            mirror=True,
            rangemode='tozero'
        ),
        showlegend=False,
        margin=dict(t=100, l=80, r=40, b=120),
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,
            font_family='Arial'
        )
    )

    # Add hover template
    fig_group_bar.update_traces(
        hovertemplate="<b>Group</b>: %{x}<br>" +
                     "<b>Incidents</b>: %{y}<br>" +
                     "<extra></extra>"
    )

    # Add subtle grid pattern
    for y in range(0, int(group_counts['Incidents'].max()) + 2):
        fig_group_bar.add_shape(
            type='line',
            x0=0,
            x1=1,
            y0=y,
            y1=y,
            xref='paper',
            yref='y',
            line=dict(
                color='rgba(189,189,189,0.2)',
                width=1
            )
        )

    # Add hover template for bar chart
    fig_group_bar.update_traces(
        hovertemplate="<b>Group</b>: %{x}<br>" +
                     "<b>Incidents</b>: %{y}<br>" +
                     "<extra></extra>"
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