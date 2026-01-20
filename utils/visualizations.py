"""
Visualization functions using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px


def plot_missing_values(df_before, df_after):
    """Plot missing values comparison"""
    missing_before = df_before.isnull().sum()
    missing_after = df_after.isnull().sum()
    
    # Only show columns with missing values
    columns_with_missing = missing_before[missing_before > 0].index.tolist()
    
    if not columns_with_missing:
        # If no missing values, create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values found in dataset",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=400)
        return fig
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Before',
        x=columns_with_missing,
        y=missing_before[columns_with_missing].values,
        marker_color='#ff7f0e'
    ))
    fig.add_trace(go.Bar(
        name='After',
        x=columns_with_missing,
        y=missing_after[columns_with_missing].values,
        marker_color='#2ca02c'
    ))
    
    fig.update_layout(
        title='Missing Values: Before vs After',
        xaxis_title='Columns',
        yaxis_title='Missing Count',
        barmode='group',
        height=400,
        xaxis={'tickangle': -45}
    )
    
    return fig


def plot_data_types(df):
    """Plot distribution of data types"""
    dtype_counts = df.dtypes.value_counts()
    
    fig = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index.astype(str),
        title="Distribution of Column Data Types",
        hole=0.3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig
