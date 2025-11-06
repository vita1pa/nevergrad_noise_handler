import pandas as pd
import numpy as np

# Load the data
print("Loading data files...")
sales_df = pd.read_csv('Sales.csv', sep='\t')
media_df = pd.read_csv('MediaInvestment.csv')

print(f"Sales data shape: {sales_df.shape}")
print(f"Sales columns: {sales_df.columns.tolist()}")
print("\nFirst few rows of Sales:")
print(sales_df.head())

print(f"\nMedia Investment data shape: {media_df.shape}")
print(f"Media columns: {media_df.columns.tolist()}")
print("\nMedia Investment data:")
print(media_df)

# Parse the Date column to extract Year and Month
print("\n" + "="*80)
print("Parsing dates from Sales data...")
sales_df['Date'] = pd.to_datetime(sales_df['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
sales_df['Year'] = sales_df['Date'].dt.year
sales_df['Month'] = sales_df['Date'].dt.month

# Check for parsing issues
print(f"Rows with valid dates: {sales_df['Date'].notna().sum()}/{len(sales_df)}")

# Filter data from 2015-07 to 2016-06
print("\n" + "="*80)
print("Filtering data from July 2015 to June 2016...")

# Create a date column for easier filtering
if 'Year' in sales_df.columns and 'Month' in sales_df.columns:
    sales_df['YearMonth'] = sales_df['Year'].astype(str) + '-' + sales_df['Month'].astype(str).str.zfill(2)
    
    # Filter sales data
    filtered_sales = sales_df[
        ((sales_df['Year'] == 2015) & (sales_df['Month'] >= 7)) |
        ((sales_df['Year'] == 2016) & (sales_df['Month'] <= 6))
    ].copy()
    
    print(f"Filtered sales data shape: {filtered_sales.shape}")
    
    # Convert GMV and Units_sold to numeric, handling any errors
    filtered_sales['GMV'] = pd.to_numeric(filtered_sales['GMV'], errors='coerce')
    filtered_sales['Units_sold'] = pd.to_numeric(filtered_sales['Units_sold'], errors='coerce')
    
    # Group by Year and Month to calculate total sales (GMV and Units_sold)
    sales_summary = filtered_sales.groupby(['Year', 'Month']).agg({
        'GMV': 'sum',
        'Units_sold': 'sum'
    }).reset_index()
    
    sales_summary.columns = ['Year', 'Month', 'Total_GMV', 'Total_Units_Sold']
    
    print("\nSales Summary by Year and Month:")
    print(sales_summary)
    
    # Merge with Media Investment data
    print("\n" + "="*80)
    print("Merging Sales with Media Investment data...")
    
    merged_data = pd.merge(
        sales_summary,
        media_df,
        on=['Year', 'Month'],
        how='inner'
    )
    
    print(f"\nMerged data shape: {merged_data.shape}")
    print("\nMerged Data (Sales + Media Investment):")
    print(merged_data)
    
    # Save the results
    output_file = 'sales_with_media_investment.csv'
    merged_data.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Calculate and display total sales
    print("\n" + "="*80)
    print("SUMMARY STATISTICS:")
    print("="*80)
    
    total_gmv = merged_data['Total_GMV'].sum()
    total_units = merged_data['Total_Units_Sold'].sum()
    total_investment = merged_data['Total Investment'].sum()
    
    print(f"\nTotal GMV (July 2015 - June 2016): {total_gmv:,.2f}")
    print(f"Total Units Sold (July 2015 - June 2016): {total_units:,.0f}")
    print(f"Total Media Investment (July 2015 - June 2016): {total_investment:,.2f}")
    
    # Show correlation with Total Investment
    print("\n" + "="*80)
    print("Correlation Analysis:")
    print("="*80)
    
    corr_gmv = merged_data[['Total_GMV', 'Total Investment']].corr().iloc[0, 1]
    corr_units = merged_data[['Total_Units_Sold', 'Total Investment']].corr().iloc[0, 1]
    
    print(f"Total GMV vs Total Investment: {corr_gmv:.4f}")
    print(f"Total Units Sold vs Total Investment: {corr_units:.4f}")
    
    # Display the merged data nicely
    print("\n" + "="*80)
    print("Monthly Data (Sales + Media Investment):")
    print("="*80)
    display_cols = ['Year', 'Month', 'Total_GMV', 'Total_Units_Sold', 'Total Investment']
    print(merged_data[display_cols].to_string(index=False))

else:
    print("Could not find 'Year' and 'Month' columns in Sales data")
    print("Available columns:", sales_df.columns.tolist())
