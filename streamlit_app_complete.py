import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and preprocess the data"""
    try:
        df = pd.read_csv('customer_shopping.csv')
        
        # Convert date column
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y', errors='coerce')
        # Fill any NaT values with alternative format
        mask = df['invoice_date'].isna()
        if mask.any():
            df.loc[mask, 'invoice_date'] = pd.to_datetime(df.loc[mask, 'invoice_date'], format='%m/%d/%Y', errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['invoice_date'])
        
        # Add derived columns
        df['month'] = df['invoice_date'].dt.month
        df['quarter'] = df['invoice_date'].dt.quarter
        df['year'] = df['invoice_date'].dt.year
        df['day_of_week'] = df['invoice_date'].dt.day_name()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def store_region_performance(df):
    """Store vs Region Performance Analysis"""
    st.header("🏪 Store vs Region Performance")
    
    if 'shopping_mall' not in df.columns:
        st.warning("Shopping Mall column not found in data")
        return
    
    # Store performance metrics
    store_performance = df.groupby('shopping_mall').agg({
        'price': ['sum', 'count', 'mean'],
        'customer_id': 'nunique',
        'quantity': 'sum'
    }).round(2)
    
    store_performance.columns = ['total_revenue', 'transaction_count', 'avg_order_value', 'unique_customers', 'total_quantity']
    store_performance = store_performance.sort_values('total_revenue', ascending=False)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stores", len(store_performance))
    with col2:
        st.metric("Total Revenue", f"${store_performance['total_revenue'].sum():,.2f}")
    with col3:
        st.metric("Total Transactions", f"{store_performance['transaction_count'].sum():,}")
    with col4:
        st.metric("Avg Revenue per Store", f"${store_performance['total_revenue'].mean():,.2f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_revenue = px.bar(
            store_performance,
            x='shopping_mall',
            y='total_revenue',
            title="Revenue by Store",
            labels={'shopping_mall': 'Store', 'total_revenue': 'Revenue ($)'},
            color='total_revenue',
            color_continuous_scale='Blues'
        )
        fig_revenue.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        fig_transactions = px.bar(
            store_performance,
            x='shopping_mall',
            y='transaction_count',
            title="Transaction Volume by Store",
            labels={'shopping_mall': 'Store', 'transaction_count': 'Transactions'},
            color='transaction_count',
            color_continuous_scale='Greens'
        )
        fig_transactions.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_transactions, use_container_width=True)
    
    # Store efficiency analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig_aov = px.bar(
            store_performance,
            x='shopping_mall',
            y='avg_order_value',
            title="Average Order Value by Store",
            labels={'shopping_mall': 'Store', 'avg_order_value': 'AOV ($)'},
            color='avg_order_value',
            color_continuous_scale='Oranges'
        )
        fig_aov.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_aov, use_container_width=True)
    
    with col2:
        fig_customers = px.bar(
            store_performance,
            x='shopping_mall',
            y='unique_customers',
            title="Customer Count by Store",
            labels={'shopping_mall': 'Store', 'unique_customers': 'Customers'},
            color='unique_customers',
            color_continuous_scale='Purples'
        )
        fig_customers.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_customers, use_container_width=True)
    
    # Performance summary table
    st.subheader("📊 Store Performance Summary")
    store_performance['revenue_rank'] = store_performance['total_revenue'].rank(ascending=False, method='dense').astype(int)
    store_performance['transaction_rank'] = store_performance['transaction_count'].rank(ascending=False, method='dense').astype(int)
    store_performance['aov_rank'] = store_performance['avg_order_value'].rank(ascending=False, method='dense').astype(int)
    
    display_df = store_performance[['total_revenue', 'transaction_count', 'avg_order_value', 'unique_customers', 'revenue_rank', 'transaction_rank', 'aov_rank']].copy()
    display_df.columns = ['Total Revenue ($)', 'Transactions', 'Avg Order Value ($)', 'Customers', 'Revenue Rank', 'Transaction Rank', 'AOV Rank']
    
    st.dataframe(display_df, use_container_width=True)

def top_customers_analysis(df):
    """Top 10% Customers Analysis"""
    st.header("👑 Top 10% Customers Analysis")
    
    # Calculate customer metrics
    customer_metrics = df.groupby('customer_id').agg({
        'price': ['sum', 'count', 'mean'],
        'invoice_date': ['min', 'max']
    }).round(2)
    
    customer_metrics.columns = ['total_revenue', 'transaction_count', 'avg_order_value', 'first_purchase', 'last_purchase']
    customer_metrics = customer_metrics.sort_values('total_revenue', ascending=False)
    
    # Top 10% customers
    top_10_percent = int(len(customer_metrics) * 0.1)
    top_customers = customer_metrics.head(top_10_percent)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Top 10% Customers", f"{len(top_customers):,}")
    with col2:
        st.metric("Top 10% Revenue", f"${top_customers['total_revenue'].sum():,.2f}")
    with col3:
        st.metric("Revenue Share", f"{(top_customers['total_revenue'].sum() / customer_metrics['total_revenue'].sum() * 100):.1f}%")
    with col4:
        st.metric("Avg Revenue (Top 10%)", f"${top_customers['total_revenue'].mean():,.2f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_top = px.bar(
            top_customers.head(20),
            x='customer_id',
            y='total_revenue',
            title="Top 20 Customers by Revenue",
            labels={'customer_id': 'Customer ID', 'total_revenue': 'Revenue ($)'},
            color='total_revenue',
            color_continuous_scale='Blues'
        )
        fig_top.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(
            top_customers.head(10),
            values='total_revenue',
            names='customer_id',
            title="Revenue Distribution - Top 10 Customers"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Top customers table
    st.subheader("📋 Top 10% Customer Details")
    top_customers['revenue_rank'] = top_customers['total_revenue'].rank(ascending=False, method='dense').astype(int)
    top_customers['revenue_share'] = (top_customers['total_revenue'] / top_customers['total_revenue'].sum() * 100).round(2)
    
    display_df = top_customers[['total_revenue', 'transaction_count', 'avg_order_value', 'revenue_rank', 'revenue_share']].copy()
    display_df.columns = ['Total Revenue ($)', 'Transactions', 'Avg Order Value ($)', 'Rank', 'Revenue Share (%)']
    
    st.dataframe(display_df, use_container_width=True)

def value_segmentation(df):
    """High vs Low-value Customer Segmentation"""
    st.header("💎 Customer Value Segmentation")
    
    # Calculate customer metrics
    customer_metrics = df.groupby('customer_id').agg({
        'price': ['sum', 'count', 'mean']
    }).round(2)
    
    customer_metrics.columns = ['total_revenue', 'transaction_count', 'avg_order_value']
    customer_metrics = customer_metrics.sort_values('total_revenue', ascending=False)
    
    # Value segmentation using quantiles
    high_threshold = customer_metrics['total_revenue'].quantile(0.8)
    medium_threshold = customer_metrics['total_revenue'].quantile(0.5)
    
    def assign_value_segment(revenue):
        if revenue >= high_threshold:
            return 'High Value'
        elif revenue >= medium_threshold:
            return 'Medium Value'
        else:
            return 'Low Value'
    
    customer_metrics['value_segment'] = customer_metrics['total_revenue'].apply(assign_value_segment)
    
    # Segment analysis
    segment_analysis = customer_metrics.groupby('value_segment').agg({
        'total_revenue': ['sum', 'count', 'mean'],
        'transaction_count': 'mean',
        'avg_order_value': 'mean'
    }).round(2)
    
    segment_analysis.columns = ['total_revenue', 'customer_count', 'avg_revenue', 'avg_transactions', 'avg_order_value']
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        high_value = customer_metrics[customer_metrics['value_segment'] == 'High Value']
        st.metric("High Value Customers", f"{len(high_value):,}", f"{len(high_value)/len(customer_metrics)*100:.1f}%")
    with col2:
        medium_value = customer_metrics[customer_metrics['value_segment'] == 'Medium Value']
        st.metric("Medium Value Customers", f"{len(medium_value):,}", f"{len(medium_value)/len(customer_metrics)*100:.1f}%")
    with col3:
        low_value = customer_metrics[customer_metrics['value_segment'] == 'Low Value']
        st.metric("Low Value Customers", f"{len(low_value):,}", f"{len(low_value)/len(customer_metrics)*100:.1f}%")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_segments = px.pie(
            segment_analysis,
            values='customer_count',
            names=segment_analysis.index,
            title="Customer Distribution by Value Segment",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_segments, use_container_width=True)
    
    with col2:
        fig_revenue = px.bar(
            segment_analysis,
            x=segment_analysis.index,
            y='total_revenue',
            title="Revenue by Customer Value Segment",
            labels={'x': 'Customer Segment', 'y': 'Revenue ($)'},
            color='total_revenue',
            color_continuous_scale='Greens'
        )
        fig_revenue.update_layout(showlegend=False)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Value segment performance table
    st.subheader("📈 Value Segment Performance Analysis")
    segment_analysis['revenue_share'] = (segment_analysis['total_revenue'] / segment_analysis['total_revenue'].sum() * 100).round(2)
    segment_analysis['customer_share'] = (segment_analysis['customer_count'] / segment_analysis['customer_count'].sum() * 100).round(2)
    
    display_df = segment_analysis[['customer_count', 'total_revenue', 'avg_revenue', 'avg_transactions', 'avg_order_value', 'revenue_share', 'customer_share']].copy()
    display_df.columns = ['Customers', 'Total Revenue ($)', 'Avg Revenue ($)', 'Avg Transactions', 'Avg Order Value ($)', 'Revenue Share (%)', 'Customer Share (%)']
    
    st.dataframe(display_df, use_container_width=True)

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">📊 Retail Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Could not load data. Please ensure customer_shopping.csv is available.")
        return
    
    # Sidebar navigation
    st.sidebar.title("📊 Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        [
            "Store vs Region Performance",
            "Top 10% Customers",
            "Customer Value Segmentation",
            "Discount Impact Analysis",
            "Seasonality Analysis",
            "Payment Method Analysis",
            "RFM Analysis",
            "Repeat Customer Analysis",
            "Category Insights",
            "Campaign Simulation"
        ]
    )
    
    # Display selected analysis
    if analysis_type == "Store vs Region Performance":
        store_region_performance(df)
    elif analysis_type == "Top 10% Customers":
        top_customers_analysis(df)
    elif analysis_type == "Customer Value Segmentation":
        value_segmentation(df)
    else:
        st.info(f"🚧 {analysis_type} analysis is coming soon! Please select another option.")

if __name__ == "__main__":
    main()
