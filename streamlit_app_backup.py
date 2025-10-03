import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
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
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration - Disabled for Streamlit Cloud deployment
API_BASE_URL = None  # Disable API calls for cloud deployment

def fetch_api_data(endpoint):
    """Fetch data from FastAPI endpoint with error handling - Disabled for cloud deployment"""
    # For Streamlit Cloud deployment, we'll use local data instead of API
    return None

def show_store_region_performance():
    """Store vs Region Performance - Compare sales volume and revenue across stores and regions"""
    st.header("🏪 Store vs Region Performance")
    
    # Load data from local files
    try:
        customer_data = pd.read_csv('customer_shopping.csv')
    except:
        st.error("Could not load customer data. Please ensure customer_shopping.csv is available.")
        return
    
    # Store Performance
    st.subheader("🏪 Store Performance Analysis")
    store_data = fetch_api_data("/insights/store-performance")
    
    if store_data:
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by store
            store_revenue = pd.DataFrame(store_data['store_revenue'])
            fig_store = px.bar(
                store_revenue,
                x='store_id',
                y='total_revenue',
                title="Revenue by Store",
                labels={'store_id': 'Store ID', 'total_revenue': 'Revenue ($)'}
            )
            fig_store.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_store, use_container_width=True)
        
        with col2:
            # Transaction count by store
            fig_transactions = px.bar(
                store_revenue,
                x='store_id',
                y='transaction_count',
                title="Transactions by Store",
                labels={'store_id': 'Store ID', 'transaction_count': 'Transaction Count'}
            )
            fig_transactions.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_transactions, use_container_width=True)
    else:
        # Fallback: Use local data for store performance
        st.info("📊 Using local data for analysis")
        
        # Calculate store performance from local data
        if 'shopping_mall' in customer_data.columns:
            store_performance = customer_data.groupby('shopping_mall').agg({
                'price': ['sum', 'count', 'mean'],
                'customer_id': 'nunique',
                'quantity': 'sum'
            }).round(2)
            
            store_performance.columns = ['total_revenue', 'transaction_count', 'avg_order_value', 'unique_customers', 'total_quantity']
            store_performance = store_performance.reset_index()
            store_performance = store_performance.sort_values('total_revenue', ascending=False)
            
            # Key Performance Metrics
            st.subheader("📊 Key Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Stores", f"{len(store_performance)}")
            with col2:
                st.metric("Total Revenue", f"${store_performance['total_revenue'].sum():,.2f}")
            with col3:
                st.metric("Total Transactions", f"{store_performance['transaction_count'].sum():,}")
            with col4:
                st.metric("Avg Revenue per Store", f"${store_performance['total_revenue'].mean():,.2f}")
            
            # Store Performance Comparison
            st.subheader("🏪 Store Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue by Store (Sorted)
                fig_revenue = px.bar(
                    store_performance,
                    x='shopping_mall',
                    y='total_revenue',
                    title="Revenue by Store (Sorted by Revenue)",
                    labels={'shopping_mall': 'Store', 'total_revenue': 'Revenue ($)'},
                    color='total_revenue',
                    color_continuous_scale='Blues'
                )
                fig_revenue.update_layout(xaxis_tickangle=45, showlegend=False)
                st.plotly_chart(fig_revenue, use_container_width=True)
            
            with col2:
                # Transaction Volume by Store (Sorted)
                fig_transactions = px.bar(
                    store_performance,
                    x='shopping_mall',
                    y='transaction_count',
                    title="Transaction Volume by Store (Sorted by Revenue)",
                    labels={'shopping_mall': 'Store', 'transaction_count': 'Transaction Count'},
                    color='transaction_count',
                    color_continuous_scale='Greens'
                )
                fig_transactions.update_layout(xaxis_tickangle=45, showlegend=False)
                st.plotly_chart(fig_transactions, use_container_width=True)
            
            # Store Efficiency Analysis
            st.subheader("⚡ Store Efficiency Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Average Order Value by Store
                fig_aov = px.bar(
                    store_performance,
                    x='shopping_mall',
                    y='avg_order_value',
                    title="Average Order Value by Store (Sorted by Revenue)",
                    labels={'shopping_mall': 'Store', 'avg_order_value': 'Average Order Value ($)'},
                    color='avg_order_value',
                    color_continuous_scale='Oranges'
                )
                fig_aov.update_layout(xaxis_tickangle=45, showlegend=False)
                st.plotly_chart(fig_aov, use_container_width=True)
            
            with col2:
                # Customer Count by Store
                fig_customers = px.bar(
                    store_performance,
                    x='shopping_mall',
                    y='unique_customers',
                    title="Customer Count by Store (Sorted by Revenue)",
                    labels={'shopping_mall': 'Store', 'unique_customers': 'Unique Customers'},
                    color='unique_customers',
                    color_continuous_scale='Purples'
                )
                fig_customers.update_layout(xaxis_tickangle=45, showlegend=False)
                st.plotly_chart(fig_customers, use_container_width=True)
            
            # Store Performance Summary Table
            st.subheader("📋 Store Performance Summary")
            
            # Add performance rankings
            store_performance['revenue_rank'] = store_performance['total_revenue'].rank(ascending=False, method='dense').astype(int)
            store_performance['transaction_rank'] = store_performance['transaction_count'].rank(ascending=False, method='dense').astype(int)
            store_performance['aov_rank'] = store_performance['avg_order_value'].rank(ascending=False, method='dense').astype(int)
            
            # Display the sorted table
            display_columns = ['shopping_mall', 'total_revenue', 'transaction_count', 'avg_order_value', 'unique_customers', 'revenue_rank']
            display_df = store_performance[display_columns].copy()
            display_df.columns = ['Store', 'Total Revenue ($)', 'Transactions', 'Avg Order Value ($)', 'Unique Customers', 'Revenue Rank']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Top and Bottom Performers
            st.subheader("🏆 Top & Bottom Performers")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🥇 Top 3 Stores by Revenue:**")
                top_stores = store_performance.head(3)
                for idx, row in top_stores.iterrows():
                    st.write(f"**{row['revenue_rank']}.** {row['shopping_mall']} - ${row['total_revenue']:,.2f}")
            
            with col2:
                st.write("**📉 Bottom 3 Stores by Revenue:**")
                bottom_stores = store_performance.tail(3)
                for idx, row in bottom_stores.iterrows():
                    st.write(f"**{row['revenue_rank']}.** {row['shopping_mall']} - ${row['total_revenue']:,.2f}")
            
            # Performance Insights
            st.subheader("💡 Performance Insights")
            
            best_store = store_performance.iloc[0]
            worst_store = store_performance.iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Best Performing Store", 
                    best_store['shopping_mall'],
                    f"${best_store['total_revenue']:,.2f}"
                )
            with col2:
                revenue_gap = best_store['total_revenue'] - worst_store['total_revenue']
                st.metric(
                    "Revenue Gap (Best vs Worst)", 
                    f"${revenue_gap:,.2f}",
                    f"{((best_store['total_revenue'] / worst_store['total_revenue'] - 1) * 100):.1f}%"
                )
            with col3:
                avg_transactions = store_performance['transaction_count'].mean()
                st.metric(
                    "Average Transactions per Store", 
                    f"{avg_transactions:,.0f}",
                    f"±{store_performance['transaction_count'].std():.0f}"
                )
        else:
            st.warning("Shopping Mall column not found in data. Showing general performance metrics.")
            
            # Show general metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Revenue", f"${customer_data['price'].sum():,.2f}")
            with col2:
                st.metric("Total Transactions", f"{len(customer_data):,}")
            with col3:
                st.metric("Unique Customers", f"{customer_data['customer_id'].nunique():,}")
            with col4:
                st.metric("Average Order Value", f"${customer_data['price'].mean():.2f}")
    

def show_customer_segmentation_rfm():
    """Customer Segmentation & RFM - Top Customers, High vs Low Value Segmentation, and RFM Analysis"""
    st.header("👥 Customer Segmentation & RFM Analysis")
    
    # Load data from local files
    try:
        customer_data = pd.read_csv('customer_shopping.csv')
    except:
        st.error("Could not load customer data. Please ensure customer_shopping.csv is available.")
        return
    
    st.info("📊 Using local data for customer segmentation and RFM analysis")
    
    # Calculate customer metrics from local data
    customer_metrics = customer_data.groupby('customer_id').agg({
        'price': ['sum', 'count', 'mean'],
        'invoice_date': ['min', 'max']
    }).round(2)
    
    customer_metrics.columns = ['total_revenue', 'transaction_count', 'avg_order_value', 'first_purchase', 'last_purchase']
    customer_metrics = customer_metrics.reset_index()
    customer_metrics = customer_metrics.sort_values('total_revenue', ascending=False)
    
    # Key Customer Metrics
    st.subheader("📊 Key Customer Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(customer_metrics):,}")
    with col2:
        st.metric("Total Revenue", f"${customer_metrics['total_revenue'].sum():,.2f}")
    with col3:
        st.metric("Avg Revenue per Customer", f"${customer_metrics['total_revenue'].mean():.2f}")
    with col4:
        st.metric("Avg Transactions per Customer", f"{customer_metrics['transaction_count'].mean():.1f}")
    
    # Top Customers Analysis
    st.subheader("👑 Top Customers Analysis")
    
    # Calculate top 10% customers
    top_10_percent = int(len(customer_metrics) * 0.1)
    top_customers = customer_metrics.head(top_10_percent)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10% customers by revenue (bar chart)
        fig_top_customers = px.bar(
            top_customers.head(20),  # Show top 20 for better visualization
            x='customer_id',
            y='total_revenue',
            title=f"Top {top_10_percent} Customers by Revenue (Top 20 Shown)",
            labels={'customer_id': 'Customer ID', 'total_revenue': 'Revenue ($)'},
            color='total_revenue',
            color_continuous_scale='Blues'
        )
        fig_top_customers.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_top_customers, use_container_width=True)
    
    with col2:
        # Revenue distribution of top customers
        fig_top_dist = px.pie(
            top_customers.head(10),  # Show top 10 for pie chart
            values='total_revenue',
            names='customer_id',
            title="Revenue Distribution - Top 10 Customers"
        )
        st.plotly_chart(fig_top_dist, use_container_width=True)
    
    # High vs Low Value Segmentation
    st.subheader("💎 High vs Low Value Customer Segmentation")
    
    # Create value-based segments with better thresholds
    customer_metrics['value_segment'] = pd.cut(
        customer_metrics['total_revenue'],
        bins=[0, customer_metrics['total_revenue'].quantile(0.5), customer_metrics['total_revenue'].quantile(0.8), float('inf')],
        labels=['Low Value', 'Medium Value', 'High Value']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer distribution by value segment
        segment_counts = customer_metrics['value_segment'].value_counts()
        fig_segments = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Distribution by Value Segment",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_segments, use_container_width=True)
    
    with col2:
        # Revenue by segment
        segment_revenue = customer_metrics.groupby('value_segment')['total_revenue'].sum().reset_index()
        fig_revenue = px.bar(
            segment_revenue,
            x='value_segment',
            y='total_revenue',
            title="Revenue by Customer Value Segment",
            labels={'value_segment': 'Customer Segment', 'total_revenue': 'Revenue ($)'},
            color='total_revenue',
            color_continuous_scale='Greens'
        )
        fig_revenue.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Value Segment Analysis
    st.subheader("📈 Value Segment Performance Analysis")
    
    segment_analysis = customer_metrics.groupby('value_segment').agg({
        'total_revenue': ['sum', 'mean', 'count'],
        'transaction_count': 'mean',
        'avg_order_value': 'mean'
    }).round(2)
    
    segment_analysis.columns = ['total_revenue', 'avg_revenue', 'customer_count', 'avg_transactions', 'avg_order_value']
    segment_analysis = segment_analysis.reset_index()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Value Customers", f"{len(customer_metrics[customer_metrics['value_segment'] == 'High Value']):,}")
    with col2:
        st.metric("Medium Value Customers", f"{len(customer_metrics[customer_metrics['value_segment'] == 'Medium Value']):,}")
    with col3:
        st.metric("Low Value Customers", f"{len(customer_metrics[customer_metrics['value_segment'] == 'Low Value']):,}")
    
    # RFM Analysis
    st.subheader("📊 RFM Analysis")
    
    # Convert invoice_date to datetime
    from datetime import datetime
    customer_data['invoice_date'] = pd.to_datetime(customer_data['invoice_date'], errors='coerce')
    customer_data = customer_data.dropna(subset=['invoice_date'])
    
    # Calculate RFM metrics
    rfm_data = customer_data.groupby('customer_id').agg({
        'invoice_date': 'max',  # Recency
        'customer_id': 'count',  # Frequency
        'price': 'sum'  # Monetary
    }).rename(columns={'customer_id': 'frequency', 'price': 'monetary'})
    
    # Calculate recency (days since last purchase)
    rfm_data['recency'] = (datetime.now() - rfm_data['invoice_date']).dt.days
    rfm_data['recency'] = rfm_data['recency'].clip(lower=0)
    
    # Create RFM scores (1-5 scale) with error handling
    try:
        rfm_data['R_score'] = pd.qcut(rfm_data['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
    except:
        rfm_data['R_score'] = pd.cut(rfm_data['recency'], 5, labels=[5,4,3,2,1])
    
    try:
        rfm_data['F_score'] = pd.qcut(rfm_data['frequency'], 5, labels=[1,2,3,4,5], duplicates='drop')
    except:
        rfm_data['F_score'] = pd.cut(rfm_data['frequency'], 5, labels=[1,2,3,4,5])
    
    try:
        rfm_data['M_score'] = pd.qcut(rfm_data['monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
    except:
        rfm_data['M_score'] = pd.cut(rfm_data['monetary'], 5, labels=[1,2,3,4,5])
    
    # Create RFM segments with comprehensive mapping
    def assign_rfm_segment(row):
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        
        try:
            r_num = int(str(r)) if pd.notna(r) else 1
            f_num = int(str(f)) if pd.notna(f) else 1
            m_num = int(str(m)) if pd.notna(m) else 1
        except:
            return 'Others'
        
        # Champions: High R, F, M
        if r_num >= 4 and f_num >= 4 and m_num >= 4:
            return 'Champions'
        # Loyal Customers: High F, M, Medium R
        elif f_num >= 4 and m_num >= 3 and r_num >= 3:
            return 'Loyal Customers'
        # Potential Loyalists: High R, Medium F, M
        elif r_num >= 4 and f_num >= 2 and m_num >= 2:
            return 'Potential Loyalists'
        # New Customers: High R, Low F, M
        elif r_num >= 4 and f_num <= 2 and m_num >= 2:
            return 'New Customers'
        # Promising: High R, Low F, Low M
        elif r_num >= 4 and f_num <= 2 and m_num <= 2:
            return 'Promising'
        # At Risk: Low R, High F, M
        elif r_num <= 2 and f_num >= 3 and m_num >= 3:
            return 'At Risk'
        # Lost Customers: Low R, F, M
        elif r_num <= 2 and f_num <= 2 and m_num <= 2:
            return 'Lost Customers'
        # Cannot Lose Them: Low R, High F, M
        elif r_num <= 2 and f_num >= 4 and m_num >= 4:
            return 'Cannot Lose Them'
        # Hibernating: Low R, F, Medium M
        elif r_num <= 2 and f_num <= 3 and m_num >= 3:
            return 'Hibernating'
        else:
            return 'Others'
    
    rfm_data['RFM_Segment'] = rfm_data.apply(assign_rfm_segment, axis=1)
    
    # RFM Analysis Results
    st.subheader("📊 RFM Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RFM segments distribution
        rfm_segment_analysis = rfm_data.groupby('RFM_Segment').agg({
            'monetary': 'sum',
            'frequency': 'sum'
        }).reset_index()
        rfm_segment_analysis = rfm_segment_analysis.sort_values('monetary', ascending=False)
        
        fig_customers = px.bar(
            rfm_segment_analysis,
            x='RFM_Segment',
            y='frequency',
            title="Customer Count by RFM Segment",
            labels={'RFM_Segment': 'RFM Segment', 'frequency': 'Customer Count'},
            color='frequency',
            color_continuous_scale='Oranges'
        )
        fig_customers.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_customers, use_container_width=True)
    
    with col2:
        # RFM segments revenue
        fig_rfm_revenue = px.bar(
            rfm_segment_analysis,
            x='RFM_Segment',
            y='monetary',
            title="Revenue by RFM Segment",
            labels={'RFM_Segment': 'RFM Segment', 'monetary': 'Revenue ($)'},
            color='monetary',
            color_continuous_scale='Purples'
        )
        fig_rfm_revenue.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_rfm_revenue, use_container_width=True)
    
    # Key Customer Segments Summary
    st.subheader("🏆 Key Customer Segments Summary")
    
    champions = rfm_data[rfm_data['RFM_Segment'] == 'Champions']
    at_risk = rfm_data[rfm_data['RFM_Segment'] == 'At Risk']
    lost_customers = rfm_data[rfm_data['RFM_Segment'] == 'Lost Customers']
    loyal_customers = rfm_data[rfm_data['RFM_Segment'] == 'Loyal Customers']
    new_customers = rfm_data[rfm_data['RFM_Segment'] == 'New Customers']
    potential_loyalists = rfm_data[rfm_data['RFM_Segment'] == 'Potential Loyalists']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Champions", f"{len(champions):,}", f"{len(champions)/len(rfm_data)*100:.1f}%")
    with col2:
        st.metric("Loyal Customers", f"{len(loyal_customers):,}", f"{len(loyal_customers)/len(rfm_data)*100:.1f}%")
    with col3:
        st.metric("New Customers", f"{len(new_customers):,}", f"{len(new_customers)/len(rfm_data)*100:.1f}%")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("At Risk", f"{len(at_risk):,}", f"{len(at_risk)/len(rfm_data)*100:.1f}%")
    with col2:
        st.metric("Lost Customers", f"{len(lost_customers):,}", f"{len(lost_customers)/len(rfm_data)*100:.1f}%")
    with col3:
        st.metric("Potential Loyalists", f"{len(potential_loyalists):,}", f"{len(potential_loyalists)/len(rfm_data)*100:.1f}%")
    
    # Complete Segment Distribution
    st.subheader("📊 Complete Segment Distribution")
    
    segment_counts = rfm_data['RFM_Segment'].value_counts()
    fig_segments = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title="Customer Distribution by RFM Segment",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_segments, use_container_width=True)
    
    # Customer Insights
    st.subheader("💡 Customer Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Top Customer Revenue", 
            f"${customer_metrics['total_revenue'].max():,.2f}",
            f"Customer: {customer_metrics.iloc[0]['customer_id']}"
        )
    with col2:
        high_value_pct = len(customer_metrics[customer_metrics['value_segment'] == 'High Value']) / len(customer_metrics) * 100
        st.metric(
            "High Value Customer %", 
            f"{high_value_pct:.1f}%",
            f"{len(customer_metrics[customer_metrics['value_segment'] == 'High Value']):,} customers"
        )
    with col3:
        champions_pct = len(champions) / len(rfm_data) * 100
        st.metric(
            "Champions %", 
            f"{champions_pct:.1f}%",
            f"{len(champions):,} customers"
        )
    
    # Customer Performance Summary Table
    st.subheader("📋 Customer Performance Summary")
    
    # Merge customer metrics with RFM data
    customer_summary = customer_metrics.merge(
        rfm_data[['customer_id', 'RFM_Segment', 'recency', 'frequency', 'monetary']], 
        on='customer_id', 
        how='left'
    )
    
    # Add performance rankings
    customer_summary['revenue_rank'] = customer_summary['total_revenue'].rank(ascending=False, method='dense').astype(int)
    customer_summary['transaction_rank'] = customer_summary['transaction_count'].rank(ascending=False, method='dense').astype(int)
    
    # Display top 20 customers
    display_columns = ['customer_id', 'total_revenue', 'transaction_count', 'avg_order_value', 'value_segment', 'RFM_Segment', 'revenue_rank']
    display_df = customer_summary[display_columns].head(20).copy()
    display_df.columns = ['Customer ID', 'Total Revenue ($)', 'Transactions', 'Avg Order Value ($)', 'Value Segment', 'RFM Segment', 'Revenue Rank']
    
    st.dataframe(display_df, use_container_width=True)

def show_profitability_analysis():
    """Profitability Analysis - Combines Financial Analysis + Discount Impact"""
    st.header("💰 Profitability Analysis")
    
    # Load data from local files
    try:
        customer_data = pd.read_csv('customer_shopping.csv')
    except:
        st.error("Could not load customer data. Please ensure customer_shopping.csv is available.")
        return
            
            with col1:
                # Customer distribution by value segment
                segment_counts = customer_metrics['value_segment'].value_counts()
                fig_segments = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title="Customer Distribution by Value Segment"
                )
                st.plotly_chart(fig_segments, use_container_width=True)
            
            with col2:
                # Revenue by segment
                segment_revenue = customer_metrics.groupby('value_segment')['total_revenue'].sum().reset_index()
                fig_revenue = px.bar(
                    segment_revenue,
                    x='value_segment',
                    y='total_revenue',
                    title="Revenue by Customer Segment",
                    labels={'value_segment': 'Customer Segment', 'total_revenue': 'Revenue ($)'}
                )
                fig_revenue.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_revenue, use_container_width=True)
            
            # ML Model Customer Segments
            if 'customer_segment' in customer_metrics.columns:
                st.markdown("**ML Model Customer Segments**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    ml_segments = customer_metrics['customer_segment'].value_counts()
                    fig_ml = px.pie(
                        values=ml_segments.values,
                        names=ml_segments.index,
                        title="ML Model Customer Segments"
                    )
                    st.plotly_chart(fig_ml, use_container_width=True)
                
                with col2:
                    ml_revenue = customer_metrics.groupby('customer_segment')['total_revenue'].sum().reset_index()
                    fig_ml_revenue = px.bar(
                        ml_revenue,
                        x='customer_segment',
                        y='total_revenue',
                        title="Revenue by ML Segment",
                        labels={'customer_segment': 'ML Segment', 'total_revenue': 'Revenue ($)'}
                    )
                    fig_ml_revenue.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_ml_revenue, use_container_width=True)
    else:
        # Fallback: Create customer segmentation from local data
        st.info("📊 Using local data for customer segmentation analysis")
        
        # Calculate customer metrics from local data
        customer_metrics = customer_data.groupby('customer_id').agg({
            'price': ['sum', 'count', 'mean'],
            'invoice_date': ['min', 'max']
        }).round(2)
        
        customer_metrics.columns = ['total_revenue', 'transaction_count', 'avg_order_value', 'first_purchase', 'last_purchase']
        customer_metrics = customer_metrics.reset_index()
        
        # Create value-based segments
        customer_metrics['value_segment'] = pd.cut(
            customer_metrics['total_revenue'],
            bins=[0, 1000, 5000, float('inf')],
            labels=['Low Value', 'Medium Value', 'High Value']
        )
        
        # Create frequency segments
        customer_metrics['frequency_segment'] = pd.cut(
            customer_metrics['transaction_count'],
            bins=[0, 2, 5, float('inf')],
            labels=['Low Frequency', 'Medium Frequency', 'High Frequency']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Value-based segmentation
            segment_counts = customer_metrics['value_segment'].value_counts()
            fig_segments = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Distribution by Value Segment"
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        
        with col2:
            # Revenue by segment
            segment_revenue = customer_metrics.groupby('value_segment')['total_revenue'].sum().reset_index()
            fig_revenue = px.bar(
                segment_revenue,
                x='value_segment',
                y='total_revenue',
                title="Revenue by Customer Segment",
                labels={'value_segment': 'Customer Segment', 'total_revenue': 'Revenue ($)'}
            )
            fig_revenue.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Show customer metrics summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(customer_metrics):,}")
        with col2:
            st.metric("High Value Customers", f"{len(customer_metrics[customer_metrics['value_segment'] == 'High Value']):,}")
        with col3:
            st.metric("Avg Revenue per Customer", f"${customer_metrics['total_revenue'].mean():.2f}")
        with col4:
            st.metric("Avg Transactions per Customer", f"{customer_metrics['transaction_count'].mean():.1f}")
    
    # RFM Analysis
    st.subheader("📊 RFM Analysis")
    rfm_data = fetch_api_data("/insights/rfm-analysis")
    
    if rfm_data:
        try:
            rfm_df = pd.read_csv('processed_data/rfm_analysis_latest.csv', index_col=0)
        except:
            rfm_df = pd.DataFrame()
        
        if not rfm_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # RFM segments distribution
                rfm_segment_analysis = rfm_df.groupby('RFM_Segment').agg({
                    'total_revenue': 'sum',
                    'transaction_count': 'sum'
                }).reset_index()
                
                fig_customers = px.bar(
                    rfm_segment_analysis,
                    x='RFM_Segment',
                    y='transaction_count',
                    title="Customer Count by RFM Segment",
                    labels={'RFM_Segment': 'RFM Segment', 'transaction_count': 'Customer Count'}
                )
                fig_customers.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_customers, use_container_width=True)
            
            with col2:
                # RFM segments revenue
                fig_rfm_revenue = px.bar(
                    rfm_segment_analysis,
                    x='RFM_Segment',
                    y='total_revenue',
                    title="Revenue by RFM Segment",
                    labels={'RFM_Segment': 'RFM Segment', 'total_revenue': 'Revenue ($)'}
                )
                fig_rfm_revenue.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_rfm_revenue, use_container_width=True)
            
            # Key segments summary
            st.subheader("🏆 Key Customer Segments Summary")
            
            # Champions
            champions = rfm_df[rfm_df['RFM_Segment'] == 'Champions']
            at_risk = rfm_df[rfm_df['RFM_Segment'] == 'At_Risk']
            lost_customers = rfm_df[rfm_df['RFM_Segment'] == 'Lost_Customers']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Champions", f"{len(champions):,}", f"{len(champions)/len(rfm_df)*100:.1f}%")
            with col2:
                st.metric("At Risk", f"{len(at_risk):,}", f"{len(at_risk)/len(rfm_df)*100:.1f}%")
            with col3:
                st.metric("Lost Customers", f"{len(lost_customers):,}", f"{len(lost_customers)/len(rfm_df)*100:.1f}%")
            with col4:
                if not customer_metrics.empty and 'value_segment' in customer_metrics.columns:
                    high_value = customer_metrics[customer_metrics['value_segment'] == 'High-Value']
                    st.metric("High-Value", f"{len(high_value):,}", f"{len(high_value)/len(customer_metrics)*100:.1f}%")
    else:
        # Fallback: Create RFM analysis from local data
        st.info("📊 Using local data for RFM analysis")
        
        # Calculate RFM metrics from local data
        from datetime import datetime
        
        # Convert invoice_date to datetime with error handling for mixed formats
        try:
            # Try different date formats
            customer_data['invoice_date'] = pd.to_datetime(customer_data['invoice_date'], format='%d/%m/%Y', errors='coerce')
            # Fill any NaT values with alternative format
            mask = customer_data['invoice_date'].isna()
            if mask.any():
                customer_data.loc[mask, 'invoice_date'] = pd.to_datetime(customer_data.loc[mask, 'invoice_date'], format='%m/%d/%Y', errors='coerce')
        except:
            # Fallback to automatic parsing
            customer_data['invoice_date'] = pd.to_datetime(customer_data['invoice_date'], errors='coerce')
        
        # Remove rows with invalid dates
        customer_data = customer_data.dropna(subset=['invoice_date'])
        
        # Check if we have valid data after date processing
        if customer_data.empty:
            st.error("No valid data available after date processing. Please check your data format.")
            return
        
        # Calculate RFM metrics
        rfm_data = customer_data.groupby('customer_id').agg({
            'invoice_date': 'max',  # Recency
            'customer_id': 'count',  # Frequency
            'price': 'sum'  # Monetary
        }).rename(columns={'customer_id': 'frequency', 'price': 'monetary'})
        
        # Calculate recency (days since last purchase)
        rfm_data['recency'] = (datetime.now() - rfm_data['invoice_date']).dt.days
        
        # Handle any negative recency values (future dates)
        rfm_data['recency'] = rfm_data['recency'].clip(lower=0)
        
        # Create RFM scores (1-5 scale) with error handling
        try:
            rfm_data['R_score'] = pd.qcut(rfm_data['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
        except:
            rfm_data['R_score'] = pd.cut(rfm_data['recency'], 5, labels=[5,4,3,2,1])
        
        try:
            rfm_data['F_score'] = pd.qcut(rfm_data['frequency'], 5, labels=[1,2,3,4,5], duplicates='drop')
        except:
            rfm_data['F_score'] = pd.cut(rfm_data['frequency'], 5, labels=[1,2,3,4,5])
        
        try:
            rfm_data['M_score'] = pd.qcut(rfm_data['monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
        except:
            rfm_data['M_score'] = pd.cut(rfm_data['monetary'], 5, labels=[1,2,3,4,5])
        
        # Create RFM segments
        rfm_data['RFM_Segment'] = rfm_data['R_score'].astype(str) + rfm_data['F_score'].astype(str) + rfm_data['M_score'].astype(str)
        
        # Define segment names with more comprehensive mapping
        def assign_rfm_segment(row):
            r, f, m = row['R_score'], row['F_score'], row['M_score']
            
            # Convert to numeric for easier comparison
            try:
                r_num = int(str(r)) if pd.notna(r) else 1
                f_num = int(str(f)) if pd.notna(f) else 1
                m_num = int(str(m)) if pd.notna(m) else 1
            except:
                return 'Others'
            
            # Champions: High R, F, M
            if r_num >= 4 and f_num >= 4 and m_num >= 4:
                return 'Champions'
            # Loyal Customers: High F, M, Medium R
            elif f_num >= 4 and m_num >= 3 and r_num >= 3:
                return 'Loyal Customers'
            # Potential Loyalists: High R, Medium F, M
            elif r_num >= 4 and f_num >= 2 and m_num >= 2:
                return 'Potential Loyalists'
            # New Customers: High R, Low F, M
            elif r_num >= 4 and f_num <= 2 and m_num >= 2:
                return 'New Customers'
            # Promising: High R, Low F, Low M
            elif r_num >= 4 and f_num <= 2 and m_num <= 2:
                return 'Promising'
            # At Risk: Low R, High F, M
            elif r_num <= 2 and f_num >= 3 and m_num >= 3:
                return 'At Risk'
            # Lost Customers: Low R, F, M
            elif r_num <= 2 and f_num <= 2 and m_num <= 2:
                return 'Lost Customers'
            # Cannot Lose Them: Low R, High F, M
            elif r_num <= 2 and f_num >= 4 and m_num >= 4:
                return 'Cannot Lose Them'
            # Hibernating: Low R, F, Medium M
            elif r_num <= 2 and f_num <= 3 and m_num >= 3:
                return 'Hibernating'
            else:
                return 'Others'
        
        rfm_data['RFM_Segment'] = rfm_data.apply(assign_rfm_segment, axis=1)
        
        # Debug information
        st.subheader("🔍 RFM Analysis Debug Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers Analyzed", f"{len(rfm_data):,}")
        with col2:
            st.metric("Unique RFM Segments", f"{rfm_data['RFM_Segment'].nunique()}")
        with col3:
            st.metric("Avg Recency (days)", f"{rfm_data['recency'].mean():.1f}")
        
        # Show RFM score distribution
        st.write("**RFM Score Distribution:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Recency Scores:", rfm_data['R_score'].value_counts().sort_index())
        with col2:
            st.write("Frequency Scores:", rfm_data['F_score'].value_counts().sort_index())
        with col3:
            st.write("Monetary Scores:", rfm_data['M_score'].value_counts().sort_index())
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RFM segments distribution
            rfm_segment_analysis = rfm_data.groupby('RFM_Segment').agg({
                'monetary': 'sum',
                'frequency': 'sum'
            }).reset_index()
            
            fig_customers = px.bar(
                rfm_segment_analysis,
                x='RFM_Segment',
                y='frequency',
                title="Customer Count by RFM Segment",
                labels={'RFM_Segment': 'RFM Segment', 'frequency': 'Customer Count'}
            )
            fig_customers.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_customers, use_container_width=True)
        
        with col2:
            # RFM segments revenue
            fig_rfm_revenue = px.bar(
                rfm_segment_analysis,
                x='RFM_Segment',
                y='monetary',
                title="Revenue by RFM Segment",
                labels={'RFM_Segment': 'RFM Segment', 'monetary': 'Revenue ($)'}
            )
            fig_rfm_revenue.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_rfm_revenue, use_container_width=True)
        
        # Key segments summary
        st.subheader("🏆 Key Customer Segments Summary")
        
        champions = rfm_data[rfm_data['RFM_Segment'] == 'Champions']
        at_risk = rfm_data[rfm_data['RFM_Segment'] == 'At Risk']
        lost_customers = rfm_data[rfm_data['RFM_Segment'] == 'Lost Customers']
        loyal_customers = rfm_data[rfm_data['RFM_Segment'] == 'Loyal Customers']
        new_customers = rfm_data[rfm_data['RFM_Segment'] == 'New Customers']
        potential_loyalists = rfm_data[rfm_data['RFM_Segment'] == 'Potential Loyalists']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Champions", f"{len(champions):,}", f"{len(champions)/len(rfm_data)*100:.1f}%")
        with col2:
            st.metric("Loyal Customers", f"{len(loyal_customers):,}", f"{len(loyal_customers)/len(rfm_data)*100:.1f}%")
        with col3:
            st.metric("New Customers", f"{len(new_customers):,}", f"{len(new_customers)/len(rfm_data)*100:.1f}%")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("At Risk", f"{len(at_risk):,}", f"{len(at_risk)/len(rfm_data)*100:.1f}%")
        with col2:
            st.metric("Lost Customers", f"{len(lost_customers):,}", f"{len(lost_customers)/len(rfm_data)*100:.1f}%")
        with col3:
            st.metric("Potential Loyalists", f"{len(potential_loyalists):,}", f"{len(potential_loyalists)/len(rfm_data)*100:.1f}%")
        
        # Show segment distribution
        st.subheader("📊 Complete Segment Distribution")
        segment_counts = rfm_data['RFM_Segment'].value_counts()
        fig_segments = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Distribution by RFM Segment"
        )
        st.plotly_chart(fig_segments, use_container_width=True)

def show_profitability_analysis():
    """Profitability Analysis - Combines Financial Analysis + Discount Impact"""
    st.header("💰 Profitability Analysis")
    
    # Load data from local files
    try:
        customer_data = pd.read_csv('customer_shopping.csv')
    except:
        st.error("Could not load customer data. Please ensure customer_shopping.csv is available.")
        return
    
    # Discount Impact Analysis
    st.subheader("💸 Discount Impact on Profitability")
    discount_data = fetch_api_data("/insights/discount-impact")
    
    if discount_data:
        # Use full width for charts
        st.subheader("📊 Revenue Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category discount impact analysis
            category_impact = discount_data['category_discount_impact']
            
            # Extract revenue data for 10% discount scenario
            revenue_data = []
            categories = []
            for category, scenarios in category_impact.items():
                if '10%_discount' in scenarios:
                    revenue_data.append(scenarios['10%_discount']['revenue'])
                    categories.append(category)
            
            fig_category = px.bar(
                x=categories,
                y=revenue_data,
                title="Revenue by Category (10% Discount)",
                labels={'x': 'Category', 'y': 'Revenue ($)'}
            )
            fig_category.update_layout(
                xaxis_tickangle=45,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_category, use_container_width=True)
        
        with col2:
            # Discount scenarios
            scenarios_df = pd.DataFrame(discount_data['discount_scenarios']).T
            fig_scenarios = px.bar(
                scenarios_df,
                x=scenarios_df.index,
                y='total_revenue',
                title="Total Revenue by Discount Scenario",
                labels={'x': 'Discount %', 'y': 'Total Revenue ($)'}
            )
            fig_scenarios.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_scenarios, use_container_width=True)
        
        # Profitability insights
        st.subheader("📈 Profitability Insights")
        recommendations = discount_data['recommendations']
        scenarios_df = pd.DataFrame(discount_data['discount_scenarios']).T
        
        # Create wider columns for better display with improved spacing
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Better text handling for optimal discount with line breaks
            optimal_discount = recommendations['optimal_discount']
            if len(optimal_discount) > 25:
                # Split long text into multiple lines
                words = optimal_discount.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) > 20:
                        lines.append(current_line)
                        current_line = word
                    else:
                        current_line += " " + word if current_line else word
                lines.append(current_line)
                optimal_discount = "\\n".join(lines)
            
            # Use markdown to display multi-line text
            st.markdown(f"**Optimal Discount:**")
            st.markdown(f"<small>{optimal_discount}</small>", unsafe_allow_html=True)
        
        with col2:
            # Show high impact categories with better formatting
            high_impact_categories = recommendations['high_impact_categories']
            if isinstance(high_impact_categories, list):
                category_count = len(high_impact_categories)
                categories_text = ", ".join(high_impact_categories[:2])  # Show first 2 categories
                if len(high_impact_categories) > 2:
                    categories_text += f" +{len(high_impact_categories)-2} more"
            else:
                category_count = 1
                categories_text = str(high_impact_categories)
            
            # Use markdown to display with better formatting
            st.markdown(f"**High Impact Categories:**")
            st.markdown(f"<small>{category_count} categories</small>", unsafe_allow_html=True)
            if len(categories_text) > 30:
                st.markdown(f"<small>{categories_text[:27]}...</small>", unsafe_allow_html=True)
            else:
                st.markdown(f"<small>{categories_text}</small>", unsafe_allow_html=True)
        
        with col3:
            # Calculate total revenue from scenarios
            total_revenue = scenarios_df['total_revenue'].sum() if 'total_revenue' in scenarios_df.columns else 0
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
        
        with col4:
            # Show average margin
            avg_margin = scenarios_df['avg_margin'].mean() if 'avg_margin' in scenarios_df.columns else 0
            st.metric("Avg Margin", f"${avg_margin:,.0f}")
        
        # Show high impact categories with detailed metrics
        if 'high_impact_categories' in recommendations:
            st.subheader("🎯 High Impact Categories Analysis")
            
            # Use expander for better space management
            with st.expander("View Detailed Category Analysis", expanded=True):
                categories_text = recommendations['high_impact_categories']
                
                if isinstance(categories_text, str):
                    # Handle string format - try to extract category names
                    st.markdown(f"**Category Analysis: {categories_text}**")
                    
                    # Try to find categories in the discount impact data
                    if 'category_discount_impact' in discount_data:
                        st.markdown("**Detailed Category Metrics:**")
                        
                        # Calculate total revenue for percentage calculation
                        total_revenue = sum(scenarios['10%_discount'].get('revenue', 0) for scenarios in discount_data['category_discount_impact'].values() if '10%_discount' in scenarios)
                        
                        for category, scenarios in discount_data['category_discount_impact'].items():
                            if '10%_discount' in scenarios:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    revenue = scenarios['10%_discount'].get('revenue', 0)
                                    st.metric("Revenue", f"${revenue:,.0f}")
                                
                                with col2:
                                    # Calculate actual customer count from the data
                                    try:
                                        # Load customer data to get actual customer count for this category
                                        customer_data = pd.read_csv('customer_shopping.csv')
                                        category_customers = len(customer_data[customer_data['category'] == category]['customer_id'].unique())
                                        st.metric("Customers", f"{category_customers:,}")
                                    except:
                                        # Fallback to API data if file not available
                                        customers = scenarios['10%_discount'].get('customers', 0)
                                        if customers == 0:
                                            # Try to get from other scenarios
                                            for scenario_name, scenario_data in scenarios.items():
                                                if scenario_data.get('customers', 0) > 0:
                                                    customers = scenario_data.get('customers', 0)
                                                    break
                                        st.metric("Customers", f"{customers:,}")
                                
                                with col3:
                                    # Calculate revenue share percentage
                                    revenue_share = (revenue / total_revenue * 100) if total_revenue > 0 else 0
                                    st.metric("Revenue Share", f"{revenue_share:.1f}%")
                                
                                st.markdown(f"*{category}* - High discount sensitivity category")
                                st.markdown("---")
                    
                elif isinstance(categories_text, list):
                    # Handle list format
                    for i, category in enumerate(categories_text, 1):
                        st.markdown(f"**{i}. {category}**")
                        
                        # Try to find this category in the discount impact data
                        if 'category_discount_impact' in discount_data and category in discount_data['category_discount_impact']:
                            scenarios = discount_data['category_discount_impact'][category]
                            
                            if '10%_discount' in scenarios:
                                # Calculate total revenue for percentage calculation
                                total_revenue = sum(scenarios['10%_discount'].get('revenue', 0) for scenarios in discount_data['category_discount_impact'].values() if '10%_discount' in scenarios)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    revenue = scenarios['10%_discount'].get('revenue', 0)
                                    st.metric("Revenue", f"${revenue:,.0f}")
                                
                                with col2:
                                    # Calculate actual customer count from the data
                                    try:
                                        # Load customer data to get actual customer count for this category
                                        customer_data = pd.read_csv('customer_shopping.csv')
                                        category_customers = len(customer_data[customer_data['category'] == category]['customer_id'].unique())
                                        st.metric("Customers", f"{category_customers:,}")
                                    except:
                                        # Fallback to API data if file not available
                                        customers = scenarios['10%_discount'].get('customers', 0)
                                        if customers == 0:
                                            # Try to get from other scenarios
                                            for scenario_name, scenario_data in scenarios.items():
                                                if scenario_data.get('customers', 0) > 0:
                                                    customers = scenario_data.get('customers', 0)
                                                    break
                                        st.metric("Customers", f"{customers:,}")
                                
                                with col3:
                                    # Calculate revenue share percentage
                                    revenue_share = (revenue / total_revenue * 100) if total_revenue > 0 else 0
                                    st.metric("Revenue Share", f"{revenue_share:.1f}%")
                                
                                st.markdown(f"*Discount Sensitivity: High* - This category shows strong response to discount strategies")
                                st.markdown("---")
                        else:
                            st.write(f"• {category} - Detailed metrics not available")
                
                else:
                    st.write(f"• {str(categories_text)}")
                
                # Show additional insights
                st.markdown("**💡 Key Insights:**")
                st.markdown("• High impact categories show strong response to discount strategies")
                st.markdown("• These categories contribute significantly to overall revenue")
                st.markdown("• Focus discount campaigns on these categories for maximum ROI")
                st.markdown("• Monitor customer acquisition and retention in these segments")
        
        # Show discount scenarios table
        st.subheader("📊 Discount Scenarios Analysis")
        st.dataframe(scenarios_df.round(2), use_container_width=True)
    else:
        # Fallback: Create profitability analysis from local data
        st.info("📊 Using local data for profitability analysis")
        
        # Revenue Analysis
        st.subheader("📊 Revenue Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by category
            category_revenue = customer_data.groupby('category')['price'].sum().reset_index()
            category_revenue = category_revenue.sort_values('price', ascending=False)
            
            fig_category = px.bar(
                category_revenue,
                x='category',
                y='price',
                title="Revenue by Category",
                labels={'category': 'Category', 'price': 'Revenue ($)'}
            )
            fig_category.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_category, use_container_width=True)
        
        with col2:
            # Revenue by shopping mall
            mall_revenue = customer_data.groupby('shopping_mall')['price'].sum().reset_index()
            mall_revenue = mall_revenue.sort_values('price', ascending=False)
            
            fig_mall = px.bar(
                mall_revenue,
                x='shopping_mall',
                y='price',
                title="Revenue by Shopping Mall",
                labels={'shopping_mall': 'Shopping Mall', 'price': 'Revenue ($)'}
            )
            fig_mall.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_mall, use_container_width=True)
        
        # Profitability Metrics
        st.subheader("💰 Profitability Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_revenue = customer_data['price'].sum()
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        with col2:
            avg_order_value = customer_data['price'].mean()
            st.metric("Average Order Value", f"${avg_order_value:.2f}")
        with col3:
            total_transactions = len(customer_data)
            st.metric("Total Transactions", f"{total_transactions:,}")
        with col4:
            unique_customers = customer_data['customer_id'].nunique()
            st.metric("Unique Customers", f"{unique_customers:,}")
        
        # Category Performance Analysis
        st.subheader("📈 Category Performance Analysis")
        
        category_analysis = customer_data.groupby('category').agg({
            'price': ['sum', 'mean', 'count'],
            'customer_id': 'nunique',
            'quantity': 'sum'
        }).round(2)
        
        category_analysis.columns = ['total_revenue', 'avg_price', 'transaction_count', 'unique_customers', 'total_quantity']
        category_analysis = category_analysis.reset_index()
        category_analysis = category_analysis.sort_values('total_revenue', ascending=False)
        
        # Top performing categories
        st.write("**Top 10 Categories by Revenue:**")
        top_categories = category_analysis.head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_top_categories = px.bar(
                top_categories,
                x='category',
                y='total_revenue',
                title="Top 10 Categories by Revenue",
                labels={'category': 'Category', 'total_revenue': 'Revenue ($)'}
            )
            fig_top_categories.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_top_categories, use_container_width=True)
        
        with col2:
            fig_avg_price = px.bar(
                top_categories,
                x='category',
                y='avg_price',
                title="Average Price by Category",
                labels={'category': 'Category', 'avg_price': 'Average Price ($)'}
            )
            fig_avg_price.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_avg_price, use_container_width=True)
        
        # Customer Value Analysis
        st.subheader("👥 Customer Value Analysis")
        
        customer_value = customer_data.groupby('customer_id').agg({
            'price': ['sum', 'count', 'mean'],
            'quantity': 'sum'
        }).round(2)
        
        customer_value.columns = ['total_spent', 'transaction_count', 'avg_order_value', 'total_quantity']
        customer_value = customer_value.reset_index()
        customer_value = customer_value.sort_values('total_spent', ascending=False)
        
        # Customer segments based on spending
        customer_value['value_segment'] = pd.cut(
            customer_value['total_spent'],
            bins=[0, 1000, 5000, 10000, float('inf')],
            labels=['Low Value', 'Medium Value', 'High Value', 'Premium']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            segment_counts = customer_value['value_segment'].value_counts()
            fig_segments = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Value Distribution"
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        
        with col2:
            segment_revenue = customer_value.groupby('value_segment')['total_spent'].sum().reset_index()
            fig_segment_revenue = px.bar(
                segment_revenue,
                x='value_segment',
                y='total_spent',
                title="Revenue by Customer Value Segment",
                labels={'value_segment': 'Customer Segment', 'total_spent': 'Total Revenue ($)'}
            )
            st.plotly_chart(fig_segment_revenue, use_container_width=True)
        
        # Discount Impact Simulation
        st.subheader("💸 Discount Impact Simulation")
        
        # Simulate different discount scenarios
        discount_scenarios = [0, 5, 10, 15, 20, 25]
        scenario_results = []
        
        for discount in discount_scenarios:
            # Calculate revenue with discount
            discounted_revenue = total_revenue * (1 - discount/100)
            # Estimate volume increase (assume 1% increase per 1% discount)
            volume_increase = 1 + (discount * 0.01)
            adjusted_revenue = discounted_revenue * volume_increase
            net_impact = adjusted_revenue - total_revenue
            
            scenario_results.append({
                'Discount %': discount,
                'Original Revenue': total_revenue,
                'Discounted Revenue': discounted_revenue,
                'Volume Adjusted Revenue': adjusted_revenue,
                'Net Impact': net_impact,
                'ROI %': (net_impact / total_revenue) * 100
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_discount_impact = px.line(
                scenario_df,
                x='Discount %',
                y='Volume Adjusted Revenue',
                title="Revenue Impact of Discounts",
                labels={'Discount %': 'Discount Percentage', 'Volume Adjusted Revenue': 'Revenue ($)'}
            )
            st.plotly_chart(fig_discount_impact, use_container_width=True)
        
        with col2:
            fig_roi = px.bar(
                scenario_df,
                x='Discount %',
                y='ROI %',
                title="ROI by Discount Level",
                labels={'Discount %': 'Discount Percentage', 'ROI %': 'ROI (%)'}
            )
            st.plotly_chart(fig_roi, use_container_width=True)
        
        # Show scenario results table
        st.subheader("📊 Discount Scenario Results")
        st.dataframe(scenario_df.round(2), use_container_width=True)
        
        # Key Insights
        st.subheader("💡 Key Insights")
        best_scenario = scenario_df.loc[scenario_df['ROI %'].idxmax()]
        st.markdown(f"**Best Discount Strategy:** {best_scenario['Discount %']}% discount with {best_scenario['ROI %']:.1f}% ROI")
        st.markdown(f"**Revenue Impact:** ${best_scenario['Net Impact']:,.2f}")
        st.markdown(f"**Adjusted Revenue:** ${best_scenario['Volume Adjusted Revenue']:,.2f}")

def show_seasonal_trend_analysis():
    """Seasonal Trend Analysis - Combines Seasonality + Repeat vs One-time"""
    st.header("📈 Seasonal Trend Analysis")
    
    # Seasonality Analysis
    st.subheader("🌍 Seasonality Analysis")
    seasonal_data = fetch_api_data("/insights/seasonality-analysis")
    
    if seasonal_data:
        # Monthly trends
        monthly_df = pd.DataFrame(seasonal_data['monthly_trends']).T
        
        # Create proper month labels
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        
        # Convert index to month names (handle both numeric and string indices)
        try:
            # Try to convert index to numeric first
            numeric_index = pd.to_numeric(monthly_df.index, errors='coerce')
            if not numeric_index.isna().any():
                monthly_df['month_name'] = numeric_index.map(month_names)
                x_axis_data = monthly_df['month_name']
            else:
                # If conversion fails, use original index
                x_axis_data = monthly_df.index
        except:
            # Fallback to original index
            x_axis_data = monthly_df.index
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_monthly = px.line(
                monthly_df,
                x=x_axis_data,
                y='total_revenue',
                title="Monthly Revenue Trend",
                labels={'x': 'Month', 'y': 'Revenue ($)'}
            )
            fig_monthly.update_layout(
                xaxis_title="Month",
                yaxis_title="Revenue ($)",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            fig_customers = px.line(
                monthly_df,
                x=x_axis_data,
                y='unique_customers',
                title="Monthly Customer Count",
                labels={'x': 'Month', 'y': 'Customers'}
            )
            fig_customers.update_layout(
                xaxis_title="Month",
                yaxis_title="Customers",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_customers, use_container_width=True)
        
        # Seasonal patterns
        st.subheader("🌍 Seasonal Patterns")
        seasonal_df = pd.DataFrame(seasonal_data['seasonal_patterns']).T
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_seasonal = px.bar(
                seasonal_df,
                x=seasonal_df.index,
                y='total_revenue',
                title="Revenue by Season",
                labels={'x': 'Season', 'y': 'Revenue ($)'}
            )
            st.plotly_chart(fig_seasonal, use_container_width=True)
        
        with col2:
            fig_weekday = px.bar(
                seasonal_df,
                x=seasonal_df.index,
                y='unique_customers',
                title="Revenue by Weekday",
                labels={'x': 'Weekday', 'y': 'Revenue ($)'}
            )
            st.plotly_chart(fig_weekday, use_container_width=True)
    
    # Repeat vs One-time Customer Analysis
    st.subheader("🔄 Repeat vs One-time Customer Analysis")
    repeat_data = fetch_api_data("/insights/repeat-vs-onetime")
    
    if repeat_data:
        # Get customer analysis data
        customer_analysis = repeat_data['customer_analysis']
        insights = repeat_data['insights']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer type distribution using available data
            if 'one_time_customers' in customer_analysis and 'high_value_customers' in customer_analysis:
                # Extract count values from nested dictionaries
                customer_types = ['One-time', 'High-Value', 'Diverse']
                customer_counts = [
                    customer_analysis.get('one_time_customers', {}).get('count', 0),
                    customer_analysis.get('high_value_customers', {}).get('count', 0),
                    customer_analysis.get('diverse_customers', {}).get('count', 0)
                ]
                
                fig_types = px.pie(
                    values=customer_counts,
                    names=customer_types,
                    title="Customer Type Distribution"
                )
                st.plotly_chart(fig_types, use_container_width=True)
            else:
                st.info("Customer type data not available in expected format")
        
        with col2:
            # Show insights instead of revenue contribution
            st.markdown("### 💡 Customer Analysis Insights")
            if 'note' in insights:
                st.write(f"**Note:** {insights['note']}")
            if 'recommendation' in insights:
                st.write(f"**Recommendation:** {insights['recommendation']}")
            if 'potential_repeat' in insights:
                st.write(f"**Potential Repeat:** {insights['potential_repeat']}")
        
        # Key metrics using available data
        st.subheader("📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            one_time_count = customer_analysis.get('one_time_customers', {}).get('count', 0)
            st.metric("One-time Customers", f"{one_time_count:,}")
        with col2:
            high_value_count = customer_analysis.get('high_value_customers', {}).get('count', 0)
            st.metric("High-Value Customers", f"{high_value_count:,}")
        with col3:
            diverse_count = customer_analysis.get('diverse_customers', {}).get('count', 0)
            st.metric("Diverse Customers", f"{diverse_count:,}")
        with col4:
            total_customers = one_time_count + high_value_count + diverse_count
            st.metric("Total Customers", f"{total_customers:,}")
        
        # Show detailed customer analysis
        st.subheader("📋 Detailed Customer Analysis")
        
        # Create a proper DataFrame with extracted values
        analysis_data = []
        for key, value in customer_analysis.items():
            if isinstance(value, dict):
                analysis_data.append({
                    'Customer Type': key.replace('_', ' ').title(),
                    'Count': value.get('count', 0),
                    'Percentage': f"{value.get('percentage', 0):.1f}%",
                    'Revenue': f"${value.get('total_revenue', 0):,.0f}",
                    'Avg Transaction': f"${value.get('avg_transaction_value', 0):,.0f}"
                })
        
        if analysis_data:
            customer_df = pd.DataFrame(analysis_data)
            st.dataframe(customer_df, use_container_width=True)
        else:
            st.info("No detailed customer analysis data available")

def show_payment_method_insights():
    """Payment Method Insights - Enhanced Payment Analysis"""
    st.header("💳 Payment Method Insights")
    
    payment_data = fetch_api_data("/insights/payment-method-analysis")
    
    if payment_data:
        col1, col2 = st.columns(2)
        
        with col1:
            # Payment method distribution
            payment_df = pd.DataFrame(payment_data['payment_distribution'])
            fig_payment = px.pie(
                payment_df,
                values='count',
                names='payment_method',
                title="Payment Method Distribution"
            )
            st.plotly_chart(fig_payment, use_container_width=True)
        
        with col2:
            # Revenue by payment method
            fig_revenue = px.bar(
                payment_df,
                x='payment_method',
                y='total_revenue',
                title="Revenue by Payment Method",
                labels={'payment_method': 'Payment Method', 'total_revenue': 'Revenue ($)'}
            )
            fig_revenue.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Payment method insights
        st.subheader("💡 Payment Method Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Most Popular", payment_data['insights']['most_popular'])
        with col2:
            st.metric("Highest Revenue", payment_data['insights']['highest_revenue'])
        with col3:
            st.metric("Avg Transaction", f"${payment_data['insights']['avg_transaction']:,.0f}")

def show_category_insights():
    """Category Insights - Enhanced Category Analysis"""
    st.header("📦 Category Insights")
    
    category_data = fetch_api_data("/insights/category-insights")
    
    if category_data:
        # Top categories with data values
        st.subheader("🏆 Top Categories")
        
        # Get category performance data for values
        category_perf = category_data['category_performance']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**By Revenue**")
            # Sort categories by revenue and show top 5 with values
            revenue_sorted = sorted(category_perf.items(), key=lambda x: x[1]['total_revenue'], reverse=True)
            for i, (category, metrics) in enumerate(revenue_sorted[:5], 1):
                revenue = metrics['total_revenue']
                avg_transaction = metrics['avg_transaction']
                st.write(f"{i}. **{category}**")
                st.write(f"   💰 Revenue: ${revenue:,.0f}")
                st.write(f"   📊 Avg Transaction: ${avg_transaction:,.0f}")
                st.write("")
        
        with col2:
            st.markdown("**By Profit Margin**")
            # Sort categories by profit margin and show top 5 with values
            margin_sorted = sorted(category_perf.items(), key=lambda x: x[1]['profit_margin'], reverse=True)
            for i, (category, metrics) in enumerate(margin_sorted[:5], 1):
                margin = metrics['profit_margin']
                profit = metrics['estimated_profit']
                st.write(f"{i}. **{category}**")
                st.write(f"   📈 Margin: {margin:.1f}%")
                st.write(f"   💵 Profit: ${profit:,.0f}")
                st.write("")
        
        with col3:
            st.markdown("**By Customer Count**")
            # Sort categories by unique customers and show top 5 with values
            customer_sorted = sorted(category_perf.items(), key=lambda x: x[1]['unique_customers'], reverse=True)
            for i, (category, metrics) in enumerate(customer_sorted[:5], 1):
                customers = metrics['unique_customers']
                revenue_per_customer = metrics['revenue_per_customer']
                st.write(f"{i}. **{category}**")
                st.write(f"   👥 Customers: {customers:,}")
                st.write(f"   💰 Revenue/Customer: ${revenue_per_customer:,.0f}")
                st.write("")
        
        # Category insights summary
        st.subheader("📈 Category Performance Summary")
        
        # Calculate total metrics
        total_revenue = sum(metrics['total_revenue'] for metrics in category_perf.values())
        total_customers = sum(metrics['unique_customers'] for metrics in category_perf.values())
        total_profit = sum(metrics['estimated_profit'] for metrics in category_perf.values())
        avg_margin = sum(metrics['profit_margin'] for metrics in category_perf.values()) / len(category_perf)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
        with col2:
            st.metric("Total Customers", f"{total_customers:,}")
        with col3:
            st.metric("Total Profit", f"${total_profit:,.0f}")
        with col4:
            st.metric("Avg Margin", f"{avg_margin:.1f}%")

def show_campaign_simulation():
    """Campaign Simulation - Marketing Campaign Analysis"""
    st.header("🎯 Campaign Simulation")
    
    campaign_data = fetch_api_data("/insights/campaign-simulation")
    
    if campaign_data:
        col1, col2 = st.columns(2)
        
        with col1:
            # Campaign ROI analysis
            roi_data = campaign_data['roi_analysis']
            fig_roi = px.bar(
                x=list(roi_data.keys()),
                y=list(roi_data.values()),
                title="Campaign ROI Analysis",
                labels={'x': 'Campaign Type', 'y': 'ROI (%)'}
            )
            st.plotly_chart(fig_roi, use_container_width=True)
        
        with col2:
            # Customer targeting effectiveness
            targeting_data = campaign_data['targeting_effectiveness']
            fig_targeting = px.pie(
                values=list(targeting_data.values()),
                names=list(targeting_data.keys()),
                title="Customer Targeting Effectiveness"
            )
            st.plotly_chart(fig_targeting, use_container_width=True)
        
        # Campaign insights
        st.subheader("💡 Campaign Insights")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best ROI", f"{campaign_data['insights']['best_roi']:.1f}%")
        with col2:
            st.metric("Target Customers", f"{campaign_data['insights']['target_customers']:,}")
        with col3:
            st.metric("Expected Revenue", f"${campaign_data['insights']['expected_revenue']:,.0f}")
        with col4:
            st.metric("Campaign Cost", f"${campaign_data['insights']['campaign_cost']:,.0f}")

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">📊 Retail Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Cloud deployment notice
    st.info("🌐 **Cloud Mode**: This dashboard is running on Streamlit Cloud with local data processing. All features are fully functional!")
    
    # Sidebar navigation
    st.sidebar.title("Dashboard Navigation")
    
    # Navigation options
    nav_options = [
        "Store/Region Performance",
        "Customer Segmentation & RFM", 
        "Profitability Analysis",
        "Seasonal Trend Analysis",
        "Payment Method Insights",
        "Category Insights",
        "Campaign Simulation"
    ]
    
    selected_tab = st.sidebar.selectbox("Select Analysis", nav_options)
    
    # Display selected tab
    if selected_tab == "Store/Region Performance":
        show_store_region_performance()
    elif selected_tab == "Customer Segmentation & RFM":
        show_customer_segmentation_rfm()
    elif selected_tab == "Profitability Analysis":
        show_profitability_analysis()
    elif selected_tab == "Seasonal Trend Analysis":
        show_seasonal_trend_analysis()
    elif selected_tab == "Payment Method Insights":
        show_payment_method_insights()
    elif selected_tab == "Category Insights":
        show_category_insights()
    elif selected_tab == "Campaign Simulation":
        show_campaign_simulation()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Retail Analytics Dashboard**")
    st.sidebar.markdown("Powered by Streamlit & FastAPI")

if __name__ == "__main__":
    main()
