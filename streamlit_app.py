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
    """Customer Segmentation - Top 10% Customers and High vs Low Value Segmentation"""
    st.header("👥 Customer Segmentation Analysis")
    
    # Load data from local files
    try:
        customer_data = pd.read_csv('customer_shopping.csv')
    except:
        st.error("Could not load customer data. Please ensure customer_shopping.csv is available.")
        return
    
    st.info("📊 Using local data for customer segmentation analysis")
    
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
    
    # Top 10% Customers Analysis
    st.subheader("👑 Top 10% Customers by Purchase Value")
    
    # Calculate top 10% customers
    top_10_percent = int(len(customer_metrics) * 0.1)
    top_customers = customer_metrics.head(top_10_percent)
    
    # Top 10% Customer Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Top 10% Customers", f"{len(top_customers):,}")
    with col2:
        st.metric("Top 10% Revenue", f"${top_customers['total_revenue'].sum():,.2f}")
    with col3:
        st.metric("Revenue Share", f"{(top_customers['total_revenue'].sum() / customer_metrics['total_revenue'].sum() * 100):.1f}%")
    with col4:
        st.metric("Avg Revenue (Top 10%)", f"${top_customers['total_revenue'].mean():,.2f}")
    
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
    
    # Top 10% Customer Details Table
    st.subheader("📋 Top 10% Customer Details")
    
    # Add rankings
    top_customers_display = top_customers.copy()
    top_customers_display['rank'] = range(1, len(top_customers_display) + 1)
    top_customers_display['revenue_share'] = (top_customers_display['total_revenue'] / customer_metrics['total_revenue'].sum() * 100).round(2)
    
    display_columns = ['rank', 'customer_id', 'total_revenue', 'transaction_count', 'avg_order_value', 'revenue_share']
    display_df = top_customers_display[display_columns].copy()
    display_df.columns = ['Rank', 'Customer ID', 'Total Revenue ($)', 'Transactions', 'Avg Order Value ($)', 'Revenue Share (%)']
    
    st.dataframe(display_df, use_container_width=True)
    
    # High vs Low Value Segmentation
    st.subheader("💎 High vs Low Value Customer Segmentation")
    
    # Create value-based segments with better thresholds
    customer_metrics['value_segment'] = pd.cut(
        customer_metrics['total_revenue'],
        bins=[0, customer_metrics['total_revenue'].quantile(0.5), customer_metrics['total_revenue'].quantile(0.8), float('inf')],
        labels=['Low Value', 'Medium Value', 'High Value']
    )
    
    # Value Segment Metrics
    col1, col2, col3 = st.columns(3)
    
    high_value_customers = customer_metrics[customer_metrics['value_segment'] == 'High Value']
    medium_value_customers = customer_metrics[customer_metrics['value_segment'] == 'Medium Value']
    low_value_customers = customer_metrics[customer_metrics['value_segment'] == 'Low Value']
    
    with col1:
        st.metric("High Value Customers", f"{len(high_value_customers):,}", f"{len(high_value_customers)/len(customer_metrics)*100:.1f}%")
    with col2:
        st.metric("Medium Value Customers", f"{len(medium_value_customers):,}", f"{len(medium_value_customers)/len(customer_metrics)*100:.1f}%")
    with col3:
        st.metric("Low Value Customers", f"{len(low_value_customers):,}", f"{len(low_value_customers)/len(customer_metrics)*100:.1f}%")
            
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
            
    # Value Segment Performance Analysis
    st.subheader("📈 Value Segment Performance Analysis")
    
    segment_analysis = customer_metrics.groupby('value_segment').agg({
        'total_revenue': ['sum', 'mean', 'count'],
        'transaction_count': 'mean',
        'avg_order_value': 'mean'
    }).round(2)
    
    segment_analysis.columns = ['total_revenue', 'avg_revenue', 'customer_count', 'avg_transactions', 'avg_order_value']
    segment_analysis = segment_analysis.reset_index()
    
    # Display segment analysis table
    st.dataframe(segment_analysis, use_container_width=True)
    
    # Customer Insights
    st.subheader("💡 Key Customer Insights")
    
    col1, col2, col3 = st.columns(3)
            
            with col1:
        st.metric(
            "Top Customer Revenue", 
            f"${customer_metrics['total_revenue'].max():,.2f}",
            f"Customer: {customer_metrics.iloc[0]['customer_id']}"
        )
            with col2:
        high_value_pct = len(high_value_customers) / len(customer_metrics) * 100
        st.metric(
            "High Value Customer %", 
            f"{high_value_pct:.1f}%",
            f"{len(high_value_customers):,} customers"
        )
    with col3:
        top_10_revenue_share = top_customers['total_revenue'].sum() / customer_metrics['total_revenue'].sum() * 100
        st.metric(
            "Top 10% Revenue Share", 
            f"{top_10_revenue_share:.1f}%",
            f"{len(top_customers):,} customers"
        )
    
    # Customer Performance Summary Table
    st.subheader("📋 Customer Performance Summary")
    
    # Add performance rankings
    customer_metrics['revenue_rank'] = customer_metrics['total_revenue'].rank(ascending=False, method='dense').astype(int)
    customer_metrics['transaction_rank'] = customer_metrics['transaction_count'].rank(ascending=False, method='dense').astype(int)
    
    # Display top 20 customers
    display_columns = ['customer_id', 'total_revenue', 'transaction_count', 'avg_order_value', 'value_segment', 'revenue_rank']
    display_df = customer_metrics[display_columns].head(20).copy()
    display_df.columns = ['Customer ID', 'Total Revenue ($)', 'Transactions', 'Avg Order Value ($)', 'Value Segment', 'Revenue Rank']
    
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
    
    st.info("📊 Using local data for profitability analysis")
    
    # Revenue Analysis
        st.subheader("📊 Revenue Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
        # Revenue by Category
        category_revenue = customer_data.groupby('category')['price'].sum().reset_index()
        category_revenue = category_revenue.sort_values('price', ascending=False)
            
            fig_category = px.bar(
            category_revenue.head(10),
            x='category',
            y='price',
            title="Revenue by Category (Top 10)",
            labels={'category': 'Category', 'price': 'Revenue ($)'},
            color='price',
            color_continuous_scale='Blues'
        )
        fig_category.update_layout(xaxis_tickangle=45, showlegend=False)
            st.plotly_chart(fig_category, use_container_width=True)
        
        with col2:
        # Revenue by Shopping Mall
        if 'shopping_mall' in customer_data.columns:
            mall_revenue = customer_data.groupby('shopping_mall')['price'].sum().reset_index()
            mall_revenue = mall_revenue.sort_values('price', ascending=False)
            
            fig_mall = px.bar(
                mall_revenue,
                x='shopping_mall',
                y='price',
                title="Revenue by Shopping Mall",
                labels={'shopping_mall': 'Shopping Mall', 'price': 'Revenue ($)'},
                color='price',
                color_continuous_scale='Greens'
            )
            fig_mall.update_layout(xaxis_tickangle=45, showlegend=False)
            st.plotly_chart(fig_mall, use_container_width=True)
    
    # Profitability Metrics
    st.subheader("💰 Profitability Metrics")
    
        col1, col2, col3, col4 = st.columns(4)
        with col1:
        st.metric("Total Revenue", f"${customer_data['price'].sum():,.2f}")
        with col2:
        st.metric("Average Order Value", f"${customer_data['price'].mean():.2f}")
        with col3:
        st.metric("Total Transactions", f"{len(customer_data):,}")
        with col4:
        st.metric("Unique Customers", f"{customer_data['customer_id'].nunique():,}")
    
    # Category Performance Analysis
    st.subheader("📈 Category Performance Analysis")
    
    col1, col2 = st.columns(2)
                                
                                with col1:
        # Top 10 Categories by Revenue
        top_categories = customer_data.groupby('category')['price'].sum().reset_index()
        top_categories = top_categories.sort_values('price', ascending=False).head(10)
        
        fig_top_cat = px.bar(
            top_categories,
            x='category',
            y='price',
            title="Top 10 Categories by Revenue",
            labels={'category': 'Category', 'price': 'Revenue ($)'},
            color='price',
            color_continuous_scale='Oranges'
        )
        fig_top_cat.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_top_cat, use_container_width=True)
                                
                                with col2:
        # Average Price by Category
        avg_price_cat = customer_data.groupby('category')['price'].mean().reset_index()
        avg_price_cat = avg_price_cat.sort_values('price', ascending=False).head(10)
        
        fig_avg_price = px.bar(
            avg_price_cat,
            x='category',
            y='price',
            title="Average Price by Category (Top 10)",
            labels={'category': 'Category', 'price': 'Average Price ($)'},
            color='price',
            color_continuous_scale='Purples'
        )
        fig_avg_price.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_avg_price, use_container_width=True)
    
    # Customer Value Analysis
    st.subheader("👥 Customer Value Analysis")
    
    # Calculate customer value segments
    customer_value = customer_data.groupby('customer_id')['price'].sum().reset_index()
    customer_value['value_segment'] = pd.cut(
        customer_value['price'],
        bins=[0, customer_value['price'].quantile(0.5), customer_value['price'].quantile(0.8), float('inf')],
        labels=['Low Value', 'Medium Value', 'High Value']
    )
        
        col1, col2 = st.columns(2)
        
        with col1:
        # Customer Value Distribution
        value_counts = customer_value['value_segment'].value_counts()
        fig_value_dist = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title="Customer Value Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_value_dist, use_container_width=True)
        
        with col2:
        # Revenue by Customer Value Segment
        value_revenue = customer_value.groupby('value_segment')['price'].sum().reset_index()
        fig_value_revenue = px.bar(
            value_revenue,
            x='value_segment',
            y='price',
            title="Revenue by Customer Value Segment",
            labels={'value_segment': 'Customer Segment', 'price': 'Revenue ($)'},
            color='price',
            color_continuous_scale='Reds'
        )
        fig_value_revenue.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_value_revenue, use_container_width=True)
    
    # Discount Impact Simulation
    st.subheader("💸 Discount Impact Simulation")
    
    # Simulate different discount scenarios
    discount_scenarios = [0, 5, 10, 15, 20, 25]
    revenue_impact = []
    
    for discount in discount_scenarios:
        # Simulate revenue with discount (assuming 1.2x volume increase per 10% discount)
        volume_multiplier = 1 + (discount / 10) * 0.2
        discounted_price = customer_data['price'] * (1 - discount / 100)
        simulated_revenue = (discounted_price * volume_multiplier).sum()
        revenue_impact.append(simulated_revenue)
    
    # Create discount impact chart
    discount_df = pd.DataFrame({
        'Discount %': discount_scenarios,
        'Revenue': revenue_impact
    })
        
        col1, col2 = st.columns(2)
        
        with col1:
        # Revenue Impact of Discounts
        fig_discount = px.line(
            discount_df,
            x='Discount %',
            y='Revenue',
            title="Revenue Impact of Discounts",
            markers=True
        )
        fig_discount.update_layout(
            xaxis_title="Discount Percentage (%)",
            yaxis_title="Revenue ($)"
        )
        st.plotly_chart(fig_discount, use_container_width=True)
        
        with col2:
        # ROI by Discount Level
        base_revenue = revenue_impact[0]
        roi_data = []
        for i, (discount, revenue) in enumerate(zip(discount_scenarios, revenue_impact)):
            if discount > 0:
                roi = ((revenue - base_revenue) / base_revenue) * 100
                roi_data.append({'Discount %': discount, 'ROI %': roi})
        
        roi_df = pd.DataFrame(roi_data)
        fig_roi = px.bar(
            roi_df,
            x='Discount %',
            y='ROI %',
            title="ROI by Discount Level",
            color='ROI %',
            color_continuous_scale='RdYlGn'
        )
        fig_roi.update_layout(
            xaxis_title="Discount Percentage (%)",
            yaxis_title="ROI (%)"
        )
        st.plotly_chart(fig_roi, use_container_width=True)
    
    # Discount Scenario Results
    st.subheader("📊 Discount Scenario Results")
    
    scenario_data = []
    for discount, revenue in zip(discount_scenarios, revenue_impact):
        if discount == 0:
            roi = 0
        else:
            roi = ((revenue - base_revenue) / base_revenue) * 100
        
        scenario_data.append({
            'Discount %': discount,
            'Revenue ($)': f"${revenue:,.2f}",
            'ROI %': f"{roi:.1f}%",
            'Volume Multiplier': f"{1 + (discount / 10) * 0.2:.2f}x"
        })
    
    scenario_df = pd.DataFrame(scenario_data)
    st.dataframe(scenario_df, use_container_width=True)
    
    # Key Insights
    st.subheader("💡 Key Insights")
    
    best_discount_idx = np.argmax(revenue_impact)
    best_discount = discount_scenarios[best_discount_idx]
    best_revenue = revenue_impact[best_discount_idx]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
        st.metric(
            "Optimal Discount", 
            f"{best_discount}%",
            f"Revenue: ${best_revenue:,.2f}"
        )
        with col2:
        max_roi = max([((revenue - base_revenue) / base_revenue) * 100 for revenue in revenue_impact[1:]])
        st.metric(
            "Maximum ROI", 
            f"{max_roi:.1f}%",
            f"At {discount_scenarios[np.argmax([((revenue - base_revenue) / base_revenue) * 100 for revenue in revenue_impact[1:]]) + 1]}% discount"
        )
        with col3:
        revenue_gain = best_revenue - base_revenue
        st.metric(
            "Revenue Gain", 
            f"${revenue_gain:,.2f}",
            f"vs No Discount"
        )

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">📊 Retail Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Cloud Mode Notice
    st.info("🌐 **Cloud Mode**: This dashboard is running on Streamlit Cloud with local data processing. All features are fully functional!")
    
    # Sidebar navigation
    st.sidebar.title("Dashboard Navigation")
    
    # Analysis selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        [
        "Store/Region Performance",
            "Customer Segmentation Analysis",
            "Profitability Analysis"
        ]
    )
    
    # Display selected analysis
    if analysis_type == "Store/Region Performance":
        show_store_region_performance()
    elif analysis_type == "Customer Segmentation Analysis":
        show_customer_segmentation_rfm()
    elif analysis_type == "Profitability Analysis":
        show_profitability_analysis()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Retail Analytics Dashboard**")
    st.sidebar.markdown("Powered by Streamlit & FastAPI")

if __name__ == "__main__":
    main()
