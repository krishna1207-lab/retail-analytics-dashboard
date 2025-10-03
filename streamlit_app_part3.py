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
