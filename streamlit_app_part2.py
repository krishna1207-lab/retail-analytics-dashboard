def discount_impact_analysis(df):
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
