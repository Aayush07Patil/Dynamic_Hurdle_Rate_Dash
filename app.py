import dash
from dash import dcc, html, Input, Output, callback, no_update
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import request, jsonify
import os
import dash_bootstrap_components as dbc
import time

# Try to import pyodbc, but provide alternative if it fails
try:
    import pyodbc
except ImportError:
    print("pyodbc not installed. Using sample data only.")
    pyodbc = None

# Initialize the Dash app
app = dash.Dash(__name__, 
                title="Airline Dynamic Hurdle Rate Dashboard", 
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose Flask server to add custom routes

# Global variables to store the last received data
current_flight_data = {
    "flight_no": "",
    "flight_date": datetime.now().date().isoformat(),
    "flight_origin": "",
    "flight_destination": "",
    "product_type": ""
}

# Global variables for tracking operations
current_request_id = None
last_processed_request_id = None

# Layout - removed input fields and display fields, full viewport layout
app.layout = html.Div([
    # Graph container with responsive layout and loading overlay
    dcc.Loading(
        id="loading-graph",
        type="circle",
        color="#119DFF",
        children=[
            html.Div(
                id="graph-container",
                style={
                    "width": "100%", 
                    "height": "100vh",  # Use viewport height
                    "padding": "0px",   # Remove padding
                    "margin": "0px"     # Remove margin
                }
            )
        ]
    ),
    
    # Hidden div to store the flight data from the .NET application
    html.Div(id="flight-data-store", style={"display": "none"}),
    
    # Hidden div to trigger callback refresh
    html.Div(id="refresh-trigger", style={"display": "none"}),
    
    # Add interval component to trigger updates
    dcc.Interval(
        id='interval-component',
        interval=1200000,  
        n_intervals=0
    )
], style={
    "width": "100%",
    "height": "100vh",  # Use full viewport height
    "padding": "0px",   # Remove padding
    "margin": "0px",    # Remove margin
    "overflow": "hidden" # Prevent scrollbars
})

def get_data(flight_no, flight_date, origin, destination, product_type, request_id):
    # Check if this request is still the current one
    if request_id != current_request_id:
        return None, None, None
        
    try:
        if not pyodbc:
            raise ImportError("pyodbc is not available")

        # Get database connection details from environment variables
        db_server = os.environ.get('DB_SERVER', '')
        db_name = os.environ.get('DB_NAME', '')
        db_user = os.environ.get('DB_USER', '')
        db_password = os.environ.get('DB_PASSWORD', '')
        
        # Check if we have all the required connection details
        if not all([db_server, db_name, db_user, db_password]):
            print("Missing database connection details. Using sample data instead...")
            raise Exception("Missing database connection details")

        # Check if this request is still current
        if request_id != current_request_id:
            return None, None, None

        conn_str = (
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={db_server};'
            f'DATABASE={db_name};'
            f'UID={db_user};'
            f'PWD={db_password};'
            f'Encrypt=yes;'
            f'TrustServerCertificate=no;'
            f'Connection Timeout=30;'
        )
        
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Convert flight_date to datetime if it's a string
        if isinstance(flight_date, str):
            try:
                flight_date = datetime.strptime(flight_date.split('T')[0], "%Y-%m-%d").date()
            except:
                flight_date = datetime.now().date()
                
        formatted_date = flight_date.strftime("%Y-%m-%d")

        # Check if this request is still current
        if request_id != current_request_id:
            cursor.close()
            conn.close()
            return None, None, None

        cursor.execute("""
            SELECT HurdleRate FROM dbo.AirlineHurdleRate
            WHERE FlightOrigin = ? AND FlightDestination = ?
        """, (origin, destination))
        static_result = cursor.fetchone()
        static_hurdle_rate = static_result[0] if static_result else None

        # Check if this request is still current
        if request_id != current_request_id:
            cursor.close()
            conn.close()
            return None, None, None

        cursor.execute("""
            SELECT Date, Dynamic_Hurdle_Rate FROM dbo.AirlineDynamicHurdleRate
            WHERE FlightID = ? AND Departure_Date = ? AND FltOrigin = ? AND FltDestination = ? AND ProductType = ?
            ORDER BY Date
        """, (flight_no, formatted_date, origin, destination, product_type))

        dynamic_rows = cursor.fetchall()
        columns = ['Date', 'Dynamic_Hurdle_Rate']
        dynamic_df = pd.DataFrame.from_records(dynamic_rows, columns=columns)
        if not dynamic_df.empty:
            dynamic_df['Dynamic_Hurdle_Rate'] = dynamic_df['Dynamic_Hurdle_Rate'].astype(float)

        # Check if this request is still current
        if request_id != current_request_id:
            cursor.close()
            conn.close()
            return None, None, None

        cursor.execute("""
            SELECT ARM.[AWBID],ARM.[AWBPrefix],ARM.[AWBNumber],ARM.[FltNumber],ARM.[FltDate],ARM.[FltOrigin],ARM.[FltDestination],
            ARM.[Wt],ARM.[UOM],ARM.[allotmentcode],ARM.[Volume],ARM.[VolumeUnit],AT.[AWBDate],AT.[SHCCodes],
            AR.[SpotRate],AR.[RatePerKg],AR.[ChargedWeight],AR.[RateClass],AP.[AppliedPercentageDisplay],
            PM.[ProductType] 
            FROM dbo.AWBRouteMaster ARM
            LEFT JOIN dbo.AWBSummaryMaster AT ON ARM.AWBPrefix = AT.AWBPrefix AND ARM.AWBNumber = AT.AWBNumber
            LEFT JOIN dbo.AWBRateMaster AR ON ARM.AWBPrefix = AR.AWBPrefix AND ARM.AWBNumber = AR.AWBNumber
            LEFT JOIN dbo.AWBProrateLog AP ON ARM.AWBPrefix = AP.AWBPrefix AND ARM.AWBNumber = AP.AWBNumber
                AND ARM.FltOrigin = AP.FlightOrigin AND ARM.FltDestination = AP.FlightDestination
            LEFT JOIN dbo.ProductTypeMaster PM ON AT.ProductType = PM.SerialNumber
            WHERE ARM.FltNumber = ? AND ARM.FltOrigin = ? AND ARM.FltDestination = ?
        """, (flight_no, origin, destination))

        awb_data = pd.DataFrame.from_records(cursor.fetchall(), columns=[col[0] for col in cursor.description])

        cursor.close()
        conn.close()
        
        # Final check if this request is still current
        if request_id != current_request_id:
            return None, None, None
            
        return static_hurdle_rate, dynamic_df, awb_data

    except Exception as e:
        print(f"Database error: {e}\nUsing sample data instead...")

        # Check if this request is still current
        if request_id != current_request_id:
            return None, None, None

        static_hurdle_rate = 0.85
        sample_dates = [(datetime.now() - timedelta(days=30 - i)).date() for i in range(30)]
        base_rate = static_hurdle_rate
        sample_dynamic_rates = [round(base_rate + (0.05 * ((i % 9) - 4) / 10), 4) for i in range(30)]
        dynamic_df = pd.DataFrame({
            'Date': sample_dates,
            'Dynamic_Hurdle_Rate': sample_dynamic_rates
        })
        awb_data = pd.DataFrame()

        return static_hurdle_rate, dynamic_df, awb_data

# API endpoint to receive data from .NET application
@server.route('/update-data', methods=['POST'])
def update_data():
    global current_flight_data, current_request_id
    
    try:
        # Generate a new request ID to cancel any in-progress operations
        current_request_id = f"req_{datetime.now().timestamp()}"
        
        # Get the data from the request
        data = request.get_json()
        
        # Update the current flight data
        current_flight_data = {
            "flight_no": data.get("flight_no", ""),
            "flight_date": data.get("flight_date", datetime.now().date().isoformat()),
            "flight_origin": data.get("flight_origin", ""),
            "flight_destination": data.get("flight_destination", ""),
            "product_type": data.get("product_type", "")
        }
        
        print(f"Received data: {current_flight_data}")
        
        return jsonify({"status": "success", "message": "Data received successfully"}), 200
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

# New API endpoint to reset data
@server.route('/reset-data', methods=['POST'])
def reset_data():
    global current_flight_data, current_request_id
    
    try:
        # Generate a new request ID to cancel any in-progress operations
        current_request_id = f"req_{datetime.now().timestamp()}"
        
        # Reset the current flight data to empty values
        current_flight_data = {
            "flight_no": "",
            "flight_date": datetime.now().date().isoformat(),
            "flight_origin": "",
            "flight_destination": "",
            "product_type": ""
        }
        
        print("Dashboard data reset successfully")
        
        return jsonify({"status": "success", "message": "Data reset successfully"}), 200
    
    except Exception as e:
        print(f"Error resetting data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

# Callback to update the graph based on stored flight data
@callback(
    Output("graph-container", "children"),
    [Input("interval-component", "n_intervals")]
)
def update_output(n_intervals):
    global current_request_id, last_processed_request_id
    
    # If no changes to process, return no update
    if current_request_id == last_processed_request_id:
        return no_update
    
    # Store the request ID we're currently processing
    request_id = current_request_id
    last_processed_request_id = request_id
    
    try:
        # Get the current flight data
        flight_no = current_flight_data["flight_no"]
        flight_date = current_flight_data["flight_date"]
        origin = current_flight_data["flight_origin"]
        destination = current_flight_data["flight_destination"]
        product_type = current_flight_data["product_type"]
        
        if not all([flight_no, flight_date, origin, destination, product_type]):
            empty_div = html.Div(
                "Waiting for flight data...", 
                style={
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "center",
                    "height": "100%",
                    "fontSize": "16px"
                }
            )
            return empty_div

        # Check if our request is still the current one
        if request_id != current_request_id:
            return no_update

        # Get data with cancellation support
        static_rate, dynamic_df, awb_data = get_data(
            flight_no, flight_date, origin, destination, product_type, request_id
        )
        
        # Check if our request was canceled or returned no data
        if request_id != current_request_id or dynamic_df is None:
            return no_update

        if dynamic_df.empty:
            empty_div = html.Div(
                "No dynamic hurdle rate data found for the given parameters.", 
                style={
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "center",
                    "height": "100%",
                    "fontSize": "16px"
                }
            )
            return empty_div

        # Check if our request is still the current one
        if request_id != current_request_id:
            return no_update

        # Convert dates to datetime if they're not already
        dynamic_df['Date'] = pd.to_datetime(dynamic_df['Date'])
        dynamic_df['FormattedDate'] = dynamic_df['Date'].dt.strftime('%d %b').str.upper()
        
        # Create the figure with the dynamic hurdle rate line
        fig = go.Figure()
        
        # Add the dynamic hurdle rate line
        fig.add_trace(go.Scatter(
            x=dynamic_df['FormattedDate'],
            y=dynamic_df['Dynamic_Hurdle_Rate'],
            mode='lines+markers',
            name='Dynamic Hurdle Rate',
            line=dict(color='green')
        ))

        # Check if our request is still the current one
        if request_id != current_request_id:
            return no_update

        # Add the static hurdle rate line if available
        if static_rate:
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                xref="paper",
                y0=static_rate,
                y1=static_rate,
                line=dict(color="red", dash="dash")
            )
            
            # Add annotation for static rate
            fig.add_annotation(
                x=1,
                xref="paper",
                y=static_rate,
                text=f"Static Rate: {static_rate:.2f} Rs",
                showarrow=True,
                ax=50,
                ay=-30
            )

        # Check if our request is still the current one
        if request_id != current_request_id:
            return no_update

        # Add AWB rate points if data is available
        if not awb_data.empty:
            # Process AWB data
            df = awb_data.copy()
            
            # Convert date columns to datetime if they're not already
            for col in ['FltDate', 'AWBDate']:
                if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Calculate days of departure
            if 'FltDate' in df.columns and 'AWBDate' in df.columns:
                df['DOD'] = (df['FltDate'] - df['AWBDate']).dt.days
                df = df[(df['DOD'] >= 0) & (df['DOD'] <= 15)]
                
                # Calculate rate
                df['Rate'] = np.where((df['SpotRate'].isna()) | (df['SpotRate'] == 0), df['RatePerKg'], df['SpotRate'])
                df['Rate'] = np.where(df['RateClass'].isin(['M', 'F']), df['Rate'] / df['ChargedWeight'], df['Rate'])
                df = df[df['Rate'] != 0]
                df = df[df['Rate'] < 300]
                df['Rate'] = df['Rate'].astype(float).round(2)

                # Map dates
                formatted_map = dynamic_df.sort_values('Date')['FormattedDate'].tolist()
                max_idx = len(formatted_map) - 1
                dod_to_date = {dod: formatted_map[max_idx - dod] for dod in range(16) if max_idx - dod >= 0}

                df['FormattedDate'] = df['DOD'].map(dod_to_date)
                df = df.dropna(subset=['FormattedDate'])
                
                # Check if our request is still the current one
                if request_id != current_request_id:
                    return no_update
                
                # Add scatter points for AWB rates
                fig.add_trace(go.Scatter(
                    x=df['FormattedDate'], 
                    y=df['Rate'], 
                    mode='markers', 
                    name='AWB Rate',
                    marker=dict(color='blue', size=8),
                    hovertext=df.apply(lambda row: f"AWB: {row['AWBPrefix']}-{row['AWBNumber']}<br>Rate: {row['Rate']} Rs", axis=1)
                ))
                
                # Update y-axis range
                max_dynamic_rate = dynamic_df['Dynamic_Hurdle_Rate'].max()
                max_actual_rate = df['Rate'].max()
                fig.update_layout(
                    yaxis=dict(range=[0, max(max_dynamic_rate, static_rate or 0, max_actual_rate or 0) * 1.1])
                )

        # Check if our request is still the current one
        if request_id != current_request_id:
            return no_update

        # Update layout to match capacity dashboard style
        fig.update_layout(
            title=dict(
                text='Dynamic Hurdle Rate',
                x=0.5,  # Center title
                y=0.98  # Position near top
            ),
            xaxis_title='Dates',
            yaxis_title='Rate (₹)',
            hovermode="x unified",
            legend=dict(
                x=1.05,        # Just outside the right side
                y=1,           # Align to top
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent background
                bordercolor='black',
                borderwidth=1
            ),
            template='plotly_white',  # White background like the capacity dashboard
            margin=dict(l=50, r=100, t=60, b=50),  # Reduced margins
            autosize=True,    # Enable autosize for responsiveness
            height=None,      # Let height be determined by container
        )
        
        return dcc.Graph(
            figure=fig,
            style={
                'height': '100%',  # Take full height of parent container
                'width': '100%'    # Take full width of parent container
            },
            config={
                'responsive': True,  # Enable responsiveness
                'displayModeBar': False  # Hide the mode bar for cleaner appearance
            }
        )
        
    except Exception as e:
        print(f"Error updating graph: {e}")
        
        error_div = html.Div(
            f"Error updating graph: {str(e)}", 
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "height": "100%",
                "fontSize": "16px",
                "color": "red"
            }
        )
        return error_div

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0',port=int(os.environ.get('PORT',8050)))