import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import sys
import subprocess
import os
os.environ["PIP_NO_BINARY"] = "pandas"  # Force source build fallback

# Check if pandas is installed
try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas==2.2.2"])
    import pandas as pd
    
# Configure page
st.set_page_config(page_title="CDD Reporting Dashboard", page_icon="📊", layout="wide")
st.title("📊 CDD Reporting Dashboard")
st.markdown("""
This app processes Community Drug Distributor (CDD) daily stock reports and provides:
- Consolidated summary tables
- Health facility performance analysis
- Data visualization
""")

# File upload section
st.header("📤 1. Upload Daily Reports")
with st.expander("Upload Files", expanded=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    day1_file = c1.file_uploader("Day 1", type="xlsx", key="day1")
    day2_file = c2.file_uploader("Day 2", type="xlsx", key="day2")
    day3_file = c3.file_uploader("Day 3", type="xlsx", key="day3")
    day4_file = c4.file_uploader("Day 4", type="xlsx", key="day4")
    day5_file = c5.file_uploader("Day 5", type="xlsx", key="day5")

# Processing function
@st.cache_data(show_spinner=False)
def process_data(files):
    """Process uploaded files and generate analysis outputs"""
    # Read and filter data
    dfs = {}
    columns = ['Date', 'state', 'lga', 'ward', 'hf', 'username', 'CDDName', 
               'CDD to BNF - SPAQ1', 'CDD to BNF - SPAQ2']
    
    for i, file in enumerate(files, 1):
        if file:
            df = pd.read_excel(file)
            dfs[f'day{i}'] = df[columns].add_suffix(f'_day{i}').rename(
                columns={f'username_day{i}': 'username'}
            )
    
    if not dfs:
        return None, None, None
    
    # Merge datasets
    merged = dfs['day1'] if 'day1' in dfs else pd.DataFrame()
    for i in range(2, 6):
        day_key = f'day{i}'
        if day_key in dfs:
            merged = pd.merge(merged, dfs[day_key], on='username', how='outer') if not merged.empty else dfs[day_key]
    
    # Fill missing values
    for col in ['hf', 'ward', 'CDDName']:
        merged[f'{col}_day1'] = merged[f'{col}_day1'].fillna(merged[f'{col}_day2']).fillna(
            merged[f'{col}_day3']).fillna(merged[f'{col}_day4'])
    
    # Calculate totals
    spaq1_cols = [c for c in merged.columns if 'SPAQ1_day' in c]
    spaq2_cols = [c for c in merged.columns if 'SPAQ2_day' in c]
    
    merged['SPAQ1 Total'] = merged[spaq1_cols].fillna(0).sum(axis=1)
    merged['SPAQ2 Total'] = merged[spaq2_cols].fillna(0).sum(axis=1)
    merged['Total Distributed'] = merged['SPAQ1 Total'] + merged['SPAQ2 Total']
    
    # Create final DF
    final_df = merged[[
        'username', 'ward_day1', 'hf_day1', 'CDDName_day1', 
        'SPAQ1 Total', 'SPAQ2 Total', 'Total Distributed'
    ]].rename(columns={
        'ward_day1': 'Ward',
        'hf_day1': 'Health_Facility',
        'CDDName_day1': 'CDD_Name'
    })
    
    # Create health facility summaries
    hf_tables = {}
    for hf, group in final_df.groupby('Health_Facility'):
        hf_total = group[['SPAQ1 Total', 'SPAQ2 Total', 'Total Distributed']].sum()
        totals_row = pd.DataFrame({
            'CDD_Name': ['TOTAL'],
            'SPAQ1 Total': [hf_total['SPAQ1 Total']],
            'SPAQ2 Total': [hf_total['SPAQ2 Total']],
            'Total Distributed': [hf_total['Total Distributed']]
        })
        hf_tables[hf] = pd.concat([group, totals_row], ignore_index=True)
    
    return final_df, hf_tables, list(hf_tables.keys())

# Process data when files are uploaded
if any([day1_file, day2_file, day3_file, day4_file, day5_file]):
    with st.spinner("Processing data..."):
        files = [day1_file, day2_file, day3_file, day4_file, day5_file]
        final_df, hf_tables, hf_list = process_data(files)
    
    if final_df is None:
        st.warning("No valid data found in uploaded files")
        st.stop()
    
    # Data Display Section
    st.header("📋 2. View Reports")
    view_option = st.radio("Select view:", 
                          ["All Facilities Summary", "Specific Facility Report"], 
                          horizontal=True)
    
    if view_option == "All Facilities Summary":
        st.subheader("All Facilities Summary")
        st.dataframe(final_df, use_container_width=True)
        
        # Summary metrics
        st.subheader("Overall Summary Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total SPAQ1 Distributed", f"{final_df['SPAQ1 Total'].sum():,}")
        c2.metric("Total SPAQ2 Distributed", f"{final_df['SPAQ2 Total'].sum():,}")
        c3.metric("Total Medicines Distributed", f"{final_df['Total Distributed'].sum():,}")
        
    else:
        selected_hf = st.selectbox("Select Health Facility", hf_list)
        st.subheader(f"Report for {selected_hf}")
        st.dataframe(hf_tables[selected_hf], use_container_width=True)
    
    # Visualization Section
    st.header("📈 3. Visualize Data")
    viz_option = st.radio("Visualization scope:",
                         ["Show All Facilities", "Show Specific Facility"],
                         horizontal=True,
                         key="viz_option")
    
    if viz_option == "Show Specific Facility":
        viz_hf = st.selectbox("Select Facility to Visualize", hf_list, key="viz_hf")
        hf_data = final_df[final_df['Health_Facility'] == viz_hf]
        
        if not hf_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data
            cdds = hf_data['CDD_Name']
            positions = np.arange(len(cdds))
            bar_width = 0.35
            
            # Plot bars
            bars1 = ax.bar(positions, hf_data['SPAQ1 Total'], bar_width, 
                          label='SPAQ1', color='#1f77b4')
            bars2 = ax.bar(positions + bar_width, hf_data['SPAQ2 Total'], bar_width,
                          label='SPAQ2', color='#ff7f0e')
            
            # Add labels and titles
            ax.set_title(f'Distribution Summary - {viz_hf}', fontsize=16)
            ax.set_xlabel('CDD Name', fontsize=12)
            ax.set_ylabel('Quantity Distributed', fontsize=12)
            ax.set_xticks(positions + bar_width / 2)
            ax.set_xticklabels(cdds, rotation=45, ha='right')
            ax.legend()
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            st.pyplot(fig)
        else:
            st.warning(f"No data available for {viz_hf}")
    
    else:  # Show All Facilities
        st.subheader("Performance Across Facilities")
        
        # Aggregate by facility
        hf_summary = final_df.groupby('Health_Facility').agg({
            'SPAQ1 Total': 'sum',
            'SPAQ2 Total': 'sum',
            'Total Distributed': 'sum'
        }).reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        positions = np.arange(len(hf_summary))
        bar_width = 0.25
        
        # Plot bars
        ax.bar(positions - bar_width, hf_summary['SPAQ1 Total'], bar_width,
              label='SPAQ1', color='#1f77b4')
        ax.bar(positions, hf_summary['SPAQ2 Total'], bar_width,
              label='SPAQ2', color='#ff7f0e')
        ax.bar(positions + bar_width, hf_summary['Total Distributed'], bar_width,
              label='Total', color='#2ca02c')
        
        # Formatting
        ax.set_title('Distribution Summary by Health Facility', fontsize=18)
        ax.set_xlabel('Health Facility', fontsize=14)
        ax.set_ylabel('Quantity Distributed', fontsize=14)
        ax.set_xticks(positions)
        ax.set_xticklabels(hf_summary['Health_Facility'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Show facility comparison table
        st.subheader("Facility Comparison")
        st.dataframe(hf_summary.sort_values('Total Distributed', ascending=False), 
                    use_container_width=True)
    
    # Download button
    st.header("💾 4. Export Results")
    if st.button("Download Full Report as Excel"):
        with BytesIO() as buffer:
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, sheet_name='All Facilities', index=False)
                for hf, df in hf_tables.items():
                    clean_name = hf[:25] + '..' if len(hf) > 25 else hf
                    df.to_excel(writer, sheet_name=clean_name, index=False)
            st.download_button(
                label="Download Excel Report",
                data=buffer.getvalue(),
                file_name="CDD_Reporting_Summary.xlsx",
                mime="application/vnd.ms-excel"
            )

else:
    st.info("Please upload daily reports to begin analysis")
    st.image("https://images.unsplash.com/photo-1586769852836-bc069f19e1b6?auto=format&fit=crop&w=1200&h=500", 
             caption="Health Data Analysis Dashboard")

# Add footer
st.markdown("---")
st.caption("CDD Reporting Dashboard v1.0 | Community Drug Distribution Analysis")
