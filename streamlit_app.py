import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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
    .stAlert {
        margin-top: 1rem;
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
    
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
    
    # Calculate store performance metrics
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
            x=store_performance.index,
            y=store_performance['total_revenue'],
            title="Revenue by Store",
            labels={'x': 'Store', 'y': 'Revenue ($)'},
            color=store_performance['total_revenue'],
            color_continuous_scale='Blues'
        )
        fig_revenue.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        fig_transactions = px.bar(
            x=store_performance.index,
            y=store_performance['transaction_count'],
            title="Transaction Volume by Store",
            labels={'x': 'Store', 'y': 'Transactions'},
            color=store_performance['transaction_count'],
            color_continuous_scale='Greens'
        )
        fig_transactions.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_transactions, use_container_width=True)
    
    # Store efficiency analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig_aov = px.bar(
            x=store_performance.index,
            y=store_performance['avg_order_value'],
            title="Average Order Value by Store",
            labels={'x': 'Store', 'y': 'AOV ($)'},
            color=store_performance['avg_order_value'],
            color_continuous_scale='Oranges'
        )
        fig_aov.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_aov, use_container_width=True)
    
    with col2:
        fig_customers = px.bar(
            x=store_performance.index,
            y=store_performance['unique_customers'],
            title="Customer Count by Store",
            labels={'x': 'Store', 'y': 'Customers'},
            color=store_performance['unique_customers'],
            color_continuous_scale='Purples'
        )
        fig_customers.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_customers, use_container_width=True)
    
    # Performance summary table
    st.subheader("📊 Store Performance Summary")
    store_performance['revenue_rank'] = store_performance['total_revenue'].rank(ascending=False, method='dense').astype(int)
    store_performance['transaction_rank'] = store_performance['transaction_count'].rank(ascending=False, method='dense').astype(int)
    store_performance['aov_rank'] = store_performance['avg_order_value'].rank(ascending=False, method='dense').astype(int)
    
    display_df = store_performance.reset_index()
    display_df = display_df[['shopping_mall', 'total_revenue', 'transaction_count', 'avg_order_value', 'unique_customers', 'revenue_rank', 'transaction_rank', 'aov_rank']]
    display_df.columns = ['Store', 'Revenue ($)', 'Transactions', 'Avg Order Value ($)', 'Customers', 'Revenue Rank', 'Transaction Rank', 'AOV Rank']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Top and bottom performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 3 Performers")
        top_3 = store_performance.head(3)
        for idx, (store, row) in enumerate(top_3.iterrows(), 1):
            st.write(f"{idx}. **{store}**: ${row['total_revenue']:,.2f} revenue, {row['transaction_count']:,} transactions")
    
    with col2:
        st.subheader("📉 Bottom 3 Performers")
        bottom_3 = store_performance.tail(3)
        for idx, (store, row) in enumerate(bottom_3.iterrows(), 1):
            st.write(f"{idx}. **{store}**: ${row['total_revenue']:,.2f} revenue, {row['transaction_count']:,} transactions")
    
    # Performance insights
    st.subheader("💡 Performance Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_store = store_performance.index[0]
        best_revenue = store_performance.iloc[0]['total_revenue']
        st.metric("Best Performing Store", f"{best_store}", f"${best_revenue:,.2f}")
    
    with col2:
        revenue_gap = store_performance.iloc[0]['total_revenue'] - store_performance.iloc[-1]['total_revenue']
        st.metric("Revenue Gap (Best vs Worst)", f"${revenue_gap:,.2f}")
    
    with col3:
        avg_transactions = store_performance['transaction_count'].mean()
        st.metric("Average Transactions per Store", f"{avg_transactions:,.0f}")

def top_customers_analysis(df):
    """Top 10% Customers Analysis"""
    st.header("👑 Top 10% Customers Analysis")
    
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
    
    # Calculate customer metrics
    customer_metrics = df.groupby('customer_id').agg({
        'price': ['sum', 'count', 'mean'],
        'quantity': 'sum'
    }).round(2)
    
    customer_metrics.columns = ['total_revenue', 'transaction_count', 'avg_order_value', 'total_quantity']
    customer_metrics = customer_metrics.sort_values('total_revenue', ascending=False)
    
    # Identify top 10% customers
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
        fig_dist = px.pie(
            values=top_customers.head(10)['total_revenue'],
            names=top_customers.head(10).index,
            title="Revenue Distribution - Top 10 Customers"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Top customer details
    st.subheader("📋 Top 10% Customer Details")
    top_customers_display = top_customers.copy()
    top_customers_display['rank'] = range(1, len(top_customers_display) + 1)
    top_customers_display['revenue_share'] = (top_customers_display['total_revenue'] / customer_metrics['total_revenue'].sum() * 100).round(2)
    
    display_cols = ['rank', 'total_revenue', 'transaction_count', 'avg_order_value', 'revenue_share']
    st.dataframe(top_customers_display[display_cols], use_container_width=True)

def value_segmentation(df):
    """High vs Low Value Customer Segmentation"""
    st.header("💎 High vs Low Value Customer Segmentation")
    
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
    
    # Calculate customer metrics
    customer_metrics = df.groupby('customer_id').agg({
        'price': ['sum', 'count', 'mean'],
        'quantity': 'sum'
    }).round(2)
    
    customer_metrics.columns = ['total_revenue', 'transaction_count', 'avg_order_value', 'total_quantity']
    
    # Create value segments based on total spend
    high_value_threshold = customer_metrics['total_revenue'].quantile(0.8)
    medium_value_threshold = customer_metrics['total_revenue'].quantile(0.5)
    
    def assign_value_segment(revenue):
        if revenue >= high_value_threshold:
            return 'High Value'
        elif revenue >= medium_value_threshold:
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
    segment_analysis['percentage'] = (segment_analysis['customer_count'] / len(customer_metrics) * 100).round(1)
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        high_value_count = segment_analysis.loc['High Value', 'customer_count']
        high_value_pct = segment_analysis.loc['High Value', 'percentage']
        st.metric("High Value Customers", f"{high_value_count:,.0f}", f"{high_value_pct:.1f}%")
    with col2:
        medium_value_count = segment_analysis.loc['Medium Value', 'customer_count']
        medium_value_pct = segment_analysis.loc['Medium Value', 'percentage']
        st.metric("Medium Value Customers", f"{medium_value_count:,.0f}", f"{medium_value_pct:.1f}%")
    with col3:
        low_value_count = segment_analysis.loc['Low Value', 'customer_count']
        low_value_pct = segment_analysis.loc['Low Value', 'percentage']
        st.metric("Low Value Customers", f"{low_value_count:,.0f}", f"{low_value_pct:.1f}%")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_dist = px.pie(
            values=segment_analysis['customer_count'],
            names=segment_analysis.index,
            title="Customer Distribution by Value Segment",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        fig_revenue = px.bar(
            x=segment_analysis.index,
            y=segment_analysis['total_revenue'],
            title="Revenue by Customer Value Segment",
            labels={'x': 'Value Segment', 'y': 'Revenue ($)'},
            color=segment_analysis['total_revenue'],
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Value segment performance analysis
    st.subheader("📊 Value Segment Performance Analysis")
    display_df = segment_analysis.reset_index()
    display_df = display_df[['value_segment', 'customer_count', 'percentage', 'total_revenue', 'avg_revenue', 'avg_transactions', 'avg_order_value']]
    display_df.columns = ['Value Segment', 'Customer Count', 'Percentage (%)', 'Total Revenue ($)', 'Avg Revenue ($)', 'Avg Transactions', 'Avg Order Value ($)']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Customer performance summary
    st.subheader("👥 Customer Performance Summary")
    customer_summary = customer_metrics.copy()
    customer_summary = customer_summary.sort_values('total_revenue', ascending=False)
    customer_summary['revenue_rank'] = customer_summary['total_revenue'].rank(ascending=False, method='dense').astype(int)
    
    display_cols = ['customer_id', 'total_revenue', 'transaction_count', 'avg_order_value', 'value_segment', 'revenue_rank']
    st.dataframe(customer_summary[display_cols].head(20), use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">🛍️ Retail Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check if 'customer_shopping.csv' exists.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        [
            "Store vs Region Performance",
            "Top 10% Customers Analysis",
            "High vs Low Value Segmentation"
        ]
    )
    
    # Display selected analysis
    if analysis_type == "Store vs Region Performance":
        store_region_performance(df)
    elif analysis_type == "Top 10% Customers Analysis":
        top_customers_analysis(df)
    elif analysis_type == "High vs Low Value Segmentation":
        value_segmentation(df)

if __name__ == "__main__":
    main()