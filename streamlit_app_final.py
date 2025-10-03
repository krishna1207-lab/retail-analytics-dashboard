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
    elif analysis_type == "Discount Impact Analysis":
        discount_impact_analysis(df)
    elif analysis_type == "Seasonality Analysis":
        seasonality_analysis(df)
    elif analysis_type == "Payment Method Analysis":
        payment_method_analysis(df)
    elif analysis_type == "RFM Analysis":
        rfm_analysis(df)
    elif analysis_type == "Repeat Customer Analysis":
        repeat_customer_analysis(df)
    elif analysis_type == "Category Insights":
        category_insights(df)
    elif analysis_type == "Campaign Simulation":
        campaign_simulation(df)
    else:
        st.info(f"🚧 {analysis_type} analysis is coming soon! Please select another option.")

if __name__ == "__main__":
    main()def discount_impact_analysis(df):
    """Discount Impact on Profitability Analysis"""
    st.header("💰 Discount Impact on Profitability")
    
    # Calculate effective margin (price - discount)
    if 'discount' in df.columns:
        df['effective_margin'] = df['price'] - df['discount']
    else:
        # Simulate discount data for demonstration
        df['discount'] = np.random.uniform(0, df['price'] * 0.3, len(df))
        df['effective_margin'] = df['price'] - df['discount']
    
    # Revenue analysis
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"${df['price'].sum():,.2f}")
    with col2:
        st.metric("Total Discounts", f"${df['discount'].sum():,.2f}")
    with col3:
        st.metric("Effective Revenue", f"${df['effective_margin'].sum():,.2f}")
    with col4:
        st.metric("Discount Rate", f"{(df['discount'].sum() / df['price'].sum() * 100):.1f}%")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_revenue = px.bar(
            df.groupby('category')['price'].sum().reset_index(),
            x='category',
            y='price',
            title="Revenue by Category",
            labels={'category': 'Category', 'price': 'Revenue ($)'},
            color='price',
            color_continuous_scale='Blues'
        )
        fig_revenue.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        fig_discount = px.bar(
            df.groupby('category')['discount'].sum().reset_index(),
            x='category',
            y='discount',
            title="Discounts by Category",
            labels={'category': 'Category', 'discount': 'Discount ($)'},
            color='discount',
            color_continuous_scale='Reds'
        )
        fig_discount.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_discount, use_container_width=True)
    
    # Discount impact simulation
    st.subheader("📊 Discount Impact Simulation")
    
    discount_levels = [0, 5, 10, 15, 20, 25, 30]
    revenue_impact = []
    
    for discount_pct in discount_levels:
        simulated_revenue = df['price'].sum() * (1 - discount_pct / 100)
        revenue_impact.append(simulated_revenue)
    
    fig_simulation = px.line(
        x=discount_levels,
        y=revenue_impact,
        title="Revenue Impact of Discounts",
        labels={'x': 'Discount %', 'y': 'Revenue ($)'},
        markers=True
    )
    st.plotly_chart(fig_simulation, use_container_width=True)

def seasonality_analysis(df):
    """Seasonality Analysis"""
    st.header("📅 Seasonality Analysis")
    
    # Monthly trends
    monthly_data = df.groupby(['year', 'month']).agg({
        'price': ['sum', 'count'],
        'customer_id': 'nunique'
    }).round(2)
    
    monthly_data.columns = ['revenue', 'transactions', 'customers']
    monthly_data = monthly_data.reset_index()
    monthly_data['month_name'] = monthly_data['month'].map({
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    })
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Months", len(monthly_data))
    with col2:
        st.metric("Peak Month", monthly_data.loc[monthly_data['revenue'].idxmax(), 'month_name'])
    with col3:
        st.metric("Peak Revenue", f"${monthly_data['revenue'].max():,.2f}")
    with col4:
        st.metric("Avg Monthly Revenue", f"${monthly_data['revenue'].mean():,.2f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_monthly = px.line(
            monthly_data,
            x='month_name',
            y='revenue',
            title="Monthly Revenue Trends",
            labels={'month_name': 'Month', 'revenue': 'Revenue ($)'},
            markers=True
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        fig_transactions = px.bar(
            monthly_data,
            x='month_name',
            y='transactions',
            title="Monthly Transaction Volume",
            labels={'month_name': 'Month', 'transactions': 'Transactions'},
            color='transactions',
            color_continuous_scale='Greens'
        )
        fig_transactions.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_transactions, use_container_width=True)
    
    # Quarterly analysis
    st.subheader("📊 Quarterly Analysis")
    quarterly_data = df.groupby(['year', 'quarter']).agg({
        'price': ['sum', 'count'],
        'customer_id': 'nunique'
    }).round(2)
    
    quarterly_data.columns = ['revenue', 'transactions', 'customers']
    quarterly_data = quarterly_data.reset_index()
    quarterly_data['quarter_name'] = 'Q' + quarterly_data['quarter'].astype(str)
    
    fig_quarterly = px.bar(
        quarterly_data,
        x='quarter_name',
        y='revenue',
        title="Quarterly Revenue Distribution",
        labels={'quarter_name': 'Quarter', 'revenue': 'Revenue ($)'},
        color='revenue',
        color_continuous_scale='Purples'
    )
    fig_quarterly.update_layout(showlegend=False)
    st.plotly_chart(fig_quarterly, use_container_width=True)

def payment_method_analysis(df):
    """Payment Method Preference Analysis"""
    st.header("💳 Payment Method Analysis")
    
    # Simulate payment method data if not available
    if 'payment_method' not in df.columns:
        payment_methods = ['Cash', 'Card', 'UPI', 'Net Banking', 'Wallet']
        df['payment_method'] = np.random.choice(payment_methods, len(df), p=[0.3, 0.4, 0.2, 0.05, 0.05])
    
    # Payment method distribution
    payment_stats = df.groupby('payment_method').agg({
        'price': ['sum', 'count', 'mean'],
        'customer_id': 'nunique'
    }).round(2)
    
    payment_stats.columns = ['total_revenue', 'transaction_count', 'avg_transaction', 'unique_customers']
    payment_stats = payment_stats.sort_values('total_revenue', ascending=False)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Most Popular", payment_stats.index[0])
    with col2:
        st.metric("Total Revenue", f"${payment_stats['total_revenue'].sum():,.2f}")
    with col3:
        st.metric("Total Transactions", f"{payment_stats['transaction_count'].sum():,}")
    with col4:
        st.metric("Avg Transaction Value", f"${payment_stats['avg_transaction'].mean():,.2f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_payment_pie = px.pie(
            payment_stats,
            values='transaction_count',
            names=payment_stats.index,
            title="Payment Method Distribution by Transactions",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_payment_pie, use_container_width=True)
    
    with col2:
        fig_payment_revenue = px.bar(
            payment_stats,
            x=payment_stats.index,
            y='total_revenue',
            title="Revenue by Payment Method",
            labels={'x': 'Payment Method', 'y': 'Revenue ($)'},
            color='total_revenue',
            color_continuous_scale='Blues'
        )
        fig_payment_revenue.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_payment_revenue, use_container_width=True)
    
    # Payment method performance table
    st.subheader("📊 Payment Method Performance")
    payment_stats['revenue_share'] = (payment_stats['total_revenue'] / payment_stats['total_revenue'].sum() * 100).round(2)
    payment_stats['transaction_share'] = (payment_stats['transaction_count'] / payment_stats['transaction_count'].sum() * 100).round(2)
    
    display_df = payment_stats[['transaction_count', 'total_revenue', 'avg_transaction', 'unique_customers', 'revenue_share', 'transaction_share']].copy()
    display_df.columns = ['Transactions', 'Total Revenue ($)', 'Avg Transaction ($)', 'Unique Customers', 'Revenue Share (%)', 'Transaction Share (%)']
    
    st.dataframe(display_df, use_container_width=True)

def rfm_analysis(df):
    """RFM Analysis"""
    st.header("📊 RFM Analysis")
    
    # Calculate RFM metrics
    rfm_data = df.groupby('customer_id').agg({
        'invoice_date': 'max',  # Recency
        'customer_id': 'count',  # Frequency
        'price': 'sum'  # Monetary
    }).rename(columns={'customer_id': 'frequency', 'price': 'monetary'})
    
    rfm_data = rfm_data.reset_index()
    
    # Calculate recency (days since last purchase)
    rfm_data['recency'] = (datetime.now() - rfm_data['invoice_date']).dt.days
    rfm_data['recency'] = rfm_data['recency'].clip(lower=0)
    
    # Create RFM scores (1-5 scale)
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
    def assign_rfm_segment(row):
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        
        try:
            r_num = int(float(str(r))) if not pd.isna(r) else 1
            f_num = int(float(str(f))) if not pd.isna(f) else 1
            m_num = int(float(str(m))) if not pd.isna(m) else 1
        except:
            return 'Others'
        
        if r_num >= 4 and f_num >= 4 and m_num >= 4:
            return 'Champions'
        elif f_num >= 4 and m_num >= 3 and r_num >= 2:
            return 'Loyal Customers'
        elif r_num >= 4 and f_num >= 2 and m_num >= 2:
            return 'Potential Loyalists'
        elif r_num >= 4 and f_num <= 2 and m_num >= 2:
            return 'New Customers'
        elif r_num <= 2 and f_num >= 3 and m_num >= 3:
            return 'At Risk'
        elif r_num <= 2 and f_num <= 2 and m_num <= 2:
            return 'Lost Customers'
        elif r_num <= 2 and f_num >= 4 and m_num >= 4:
            return 'Cannot Lose Them'
        else:
            return 'Others'
    
    rfm_data['RFM_Segment'] = rfm_data.apply(assign_rfm_segment, axis=1)
    
    # RFM Analysis Results
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    # Key RFM Segments Summary
    st.subheader("🏆 Key RFM Segments Summary")
    
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
    
    # Complete RFM Segment Distribution
    st.subheader("📊 Complete RFM Segment Distribution")
    
    segment_counts = rfm_data['RFM_Segment'].value_counts()
    fig_segments = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title="Customer Distribution by RFM Segment",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_segments, use_container_width=True)
def repeat_customer_analysis(df):
    """Repeat Customer vs One-time Analysis"""
    st.header("🔄 Repeat Customer Analysis")
    
    # Calculate customer transaction counts
    customer_transaction_counts = df.groupby('customer_id').size().reset_index(name='transaction_count')
    
    # Classify customers
    repeat_customers = customer_transaction_counts[customer_transaction_counts['transaction_count'] > 1]
    one_time_customers = customer_transaction_counts[customer_transaction_counts['transaction_count'] == 1]
    
    # Calculate sales contribution
    repeat_customer_ids = repeat_customers['customer_id'].tolist()
    one_time_customer_ids = one_time_customers['customer_id'].tolist()
    
    repeat_customer_sales = df[df['customer_id'].isin(repeat_customer_ids)]['price'].sum()
    one_time_customer_sales = df[df['customer_id'].isin(one_time_customer_ids)]['price'].sum()
    total_sales = df['price'].sum()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Repeat Customers", f"{len(repeat_customers):,}", f"{len(repeat_customers)/len(customer_transaction_counts)*100:.1f}%")
    with col2:
        st.metric("One-time Customers", f"{len(one_time_customers):,}", f"{len(one_time_customers)/len(customer_transaction_counts)*100:.1f}%")
    with col3:
        st.metric("Repeat Customer Sales", f"${repeat_customer_sales:,.2f}", f"{repeat_customer_sales/total_sales*100:.1f}%")
    with col4:
        st.metric("One-time Customer Sales", f"${one_time_customer_sales:,.2f}", f"{one_time_customer_sales/total_sales*100:.1f}%")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        customer_types = ['Repeat Customers', 'One-time Customers']
        customer_counts = [len(repeat_customers), len(one_time_customers)]
        
        fig_customer_types = px.pie(
            values=customer_counts,
            names=customer_types,
            title="Customer Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_customer_types, use_container_width=True)
    
    with col2:
        sales_contribution = [repeat_customer_sales, one_time_customer_sales]
        
        fig_sales_contrib = px.bar(
            x=customer_types,
            y=sales_contribution,
            title="Sales Contribution by Customer Type",
            labels={'x': 'Customer Type', 'y': 'Sales ($)'},
            color=sales_contribution,
            color_continuous_scale='Blues'
        )
        fig_sales_contrib.update_layout(showlegend=False)
        st.plotly_chart(fig_sales_contrib, use_container_width=True)
    
    # Repeat Customer Analysis
    if len(repeat_customers) > 0:
        st.subheader("📈 Repeat Customer Deep Dive")
        
        repeat_customer_metrics = df[df['customer_id'].isin(repeat_customer_ids)].groupby('customer_id').agg({
            'price': ['sum', 'count', 'mean'],
            'invoice_date': ['min', 'max']
        }).round(2)
        
        repeat_customer_metrics.columns = ['total_revenue', 'transaction_count', 'avg_order_value', 'first_purchase', 'last_purchase']
        repeat_customer_metrics = repeat_customer_metrics.reset_index()
        repeat_customer_metrics = repeat_customer_metrics.sort_values('total_revenue', ascending=False)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Transactions per Repeat Customer", f"{repeat_customer_metrics['transaction_count'].mean():.1f}")
        with col2:
            st.metric("Avg Revenue per Repeat Customer", f"${repeat_customer_metrics['total_revenue'].mean():.2f}")
        with col3:
            st.metric("Avg Order Value (Repeat)", f"${repeat_customer_metrics['avg_order_value'].mean():.2f}")
        
        # Top Repeat Customers
        st.subheader("🏆 Top Repeat Customers")
        
        top_repeat_customers = repeat_customer_metrics.head(10)
        fig_top_repeat = px.bar(
            top_repeat_customers,
            x='customer_id',
            y='total_revenue',
            title="Top 10 Repeat Customers by Revenue",
            labels={'customer_id': 'Customer ID', 'total_revenue': 'Revenue ($)'},
            color='total_revenue',
            color_continuous_scale='Greens'
        )
        fig_top_repeat.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_top_repeat, use_container_width=True)

def category_insights(df):
    """Category-wise Insights"""
    st.header("📦 Category Insights")
    
    # Category analysis
    category_analysis = df.groupby('category').agg({
        'price': ['sum', 'count', 'mean'],
        'customer_id': 'nunique',
        'quantity': 'sum'
    }).round(2)
    
    category_analysis.columns = ['total_revenue', 'transaction_count', 'avg_price', 'unique_customers', 'total_quantity']
    category_analysis = category_analysis.sort_values('total_revenue', ascending=False)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Categories", len(category_analysis))
    with col2:
        st.metric("Top Category", category_analysis.index[0])
    with col3:
        st.metric("Top Category Revenue", f"${category_analysis.iloc[0]['total_revenue']:,.2f}")
    with col4:
        st.metric("Avg Revenue per Category", f"${category_analysis['total_revenue'].mean():,.2f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_revenue = px.bar(
            category_analysis.head(10),
            x=category_analysis.head(10).index,
            y='total_revenue',
            title="Top 10 Categories by Revenue",
            labels={'x': 'Category', 'y': 'Revenue ($)'},
            color='total_revenue',
            color_continuous_scale='Blues'
        )
        fig_revenue.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        fig_transactions = px.bar(
            category_analysis.head(10),
            x=category_analysis.head(10).index,
            y='transaction_count',
            title="Top 10 Categories by Transactions",
            labels={'x': 'Category', 'y': 'Transactions'},
            color='transaction_count',
            color_continuous_scale='Greens'
        )
        fig_transactions.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_transactions, use_container_width=True)
    
    # Category profitability analysis
    st.subheader("💰 Category Profitability Analysis")
    
    category_analysis['revenue_share'] = (category_analysis['total_revenue'] / category_analysis['total_revenue'].sum() * 100).round(2)
    category_analysis['transaction_share'] = (category_analysis['transaction_count'] / category_analysis['transaction_count'].sum() * 100).round(2)
    category_analysis['customer_share'] = (category_analysis['unique_customers'] / category_analysis['unique_customers'].sum() * 100).round(2)
    
    # Top profitable categories
    top_categories = category_analysis.head(10)
    
    fig_profitability = px.scatter(
        top_categories,
        x='transaction_count',
        y='total_revenue',
        size='avg_price',
        color='unique_customers',
        hover_name=top_categories.index,
        title="Category Profitability Matrix (Size = Avg Price, Color = Customers)",
        labels={'transaction_count': 'Transactions', 'total_revenue': 'Revenue ($)', 'unique_customers': 'Customers'},
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_profitability, use_container_width=True)
    
    # Category performance table
    st.subheader("📊 Category Performance Summary")
    display_df = category_analysis[['total_revenue', 'transaction_count', 'avg_price', 'unique_customers', 'revenue_share', 'transaction_share', 'customer_share']].copy()
    display_df.columns = ['Total Revenue ($)', 'Transactions', 'Avg Price ($)', 'Customers', 'Revenue Share (%)', 'Transaction Share (%)', 'Customer Share (%)']
    
    st.dataframe(display_df, use_container_width=True)

def campaign_simulation(df):
    """Campaign Simulation"""
    st.header("🎯 Campaign Simulation")
    
    # Calculate customer metrics
    customer_metrics = df.groupby('customer_id').agg({
        'price': ['sum', 'count', 'mean']
    }).round(2)
    
    customer_metrics.columns = ['total_revenue', 'transaction_count', 'avg_order_value']
    customer_metrics = customer_metrics.sort_values('total_revenue', ascending=False)
    
    # High-value customers (top 20%)
    high_value_threshold = customer_metrics['total_revenue'].quantile(0.8)
    high_value_customers = customer_metrics[customer_metrics['total_revenue'] >= high_value_threshold]
    
    # Campaign parameters
    st.subheader("📊 Campaign Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        discount_rate = st.slider("Discount Rate (%)", 0, 50, 10)
    with col2:
        target_customers = st.slider("Target Customers", 100, len(high_value_customers), min(1000, len(high_value_customers)))
    with col3:
        expected_response_rate = st.slider("Expected Response Rate (%)", 10, 100, 30)
    
    # Campaign simulation
    st.subheader("🎯 Campaign Simulation Results")
    
    # Calculate campaign metrics
    target_customer_data = high_value_customers.head(target_customers)
    base_revenue = target_customer_data['total_revenue'].sum()
    
    # Simulate campaign impact
    response_customers = int(target_customers * expected_response_rate / 100)
    campaign_revenue = base_revenue * (1 - discount_rate / 100)
    campaign_cost = base_revenue * discount_rate / 100
    campaign_profit = campaign_revenue - campaign_cost
    
    # ROI calculation
    if campaign_cost > 0:
        roi = (campaign_profit / campaign_cost) * 100
    else:
        roi = 0
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Target Customers", f"{target_customers:,}")
    with col2:
        st.metric("Expected Responses", f"{response_customers:,}")
    with col3:
        st.metric("Campaign Revenue", f"${campaign_revenue:,.2f}")
    with col4:
        st.metric("Campaign Cost", f"${campaign_cost:,.2f}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Net Profit", f"${campaign_profit:,.2f}")
    with col2:
        st.metric("ROI", f"{roi:.1f}%")
    with col3:
        st.metric("Revenue per Customer", f"${campaign_revenue/response_customers:,.2f}")
    with col4:
        st.metric("Cost per Customer", f"${campaign_cost/response_customers:,.2f}")
    
    # Campaign effectiveness visualization
    st.subheader("📈 Campaign Effectiveness Analysis")
    
    # Simulate different discount rates
    discount_rates = [0, 5, 10, 15, 20, 25, 30]
    campaign_results = []
    
    for rate in discount_rates:
        sim_revenue = base_revenue * (1 - rate / 100)
        sim_cost = base_revenue * rate / 100
        sim_profit = sim_revenue - sim_cost
        sim_roi = (sim_profit / sim_cost * 100) if sim_cost > 0 else 0
        
        campaign_results.append({
            'discount_rate': rate,
            'revenue': sim_revenue,
            'cost': sim_cost,
            'profit': sim_profit,
            'roi': sim_roi
        })
    
    campaign_df = pd.DataFrame(campaign_results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_revenue = px.line(
            campaign_df,
            x='discount_rate',
            y='revenue',
            title="Revenue vs Discount Rate",
            labels={'discount_rate': 'Discount Rate (%)', 'revenue': 'Revenue ($)'},
            markers=True
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        fig_roi = px.line(
            campaign_df,
            x='discount_rate',
            y='roi',
            title="ROI vs Discount Rate",
            labels={'discount_rate': 'Discount Rate (%)', 'roi': 'ROI (%)'},
            markers=True
        )
        st.plotly_chart(fig_roi, use_container_width=True)
    
    # Campaign recommendations
    st.subheader("💡 Campaign Recommendations")
    
    best_roi_idx = campaign_df['roi'].idxmax()
    best_discount = campaign_df.loc[best_roi_idx, 'discount_rate']
    best_roi = campaign_df.loc[best_roi_idx, 'roi']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"🎯 **Optimal Discount Rate**: {best_discount}%")
        st.success(f"📈 **Expected ROI**: {best_roi:.1f}%")
        st.success(f"💰 **Expected Profit**: ${campaign_df.loc[best_roi_idx, 'profit']:,.2f}")
    
    with col2:
        st.info(f"👥 **Target**: {target_customers:,} high-value customers")
        st.info(f"📊 **Response Rate**: {expected_response_rate}%")
        st.info(f"💵 **Revenue Impact**: ${campaign_df.loc[best_roi_idx, 'revenue']:,.2f}")
    
    # Campaign summary table
    st.subheader("📋 Campaign Summary")
    st.dataframe(campaign_df, use_container_width=True)
