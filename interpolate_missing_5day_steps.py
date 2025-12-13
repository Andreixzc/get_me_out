import pandas as pd
import numpy as np
import glob
import os
import json
from datetime import datetime, timedelta
from processing_metrics import ProcessingMetrics

def interpolate_5day_gaps():
    # Setup directories
    input_dir = "daily_snapshots"
    output_dir = "daily_snapshots_5day"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Load existing metrics if available
    metrics_path = os.path.join(input_dir, 'processing_metrics.json')
    metrics = ProcessingMetrics()
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            existing_metrics = json.load(f)
            metrics.total_grid_points = existing_metrics.get('total_grid_points', 0)
            metrics.water_points = existing_metrics.get('water_points_per_timestep', {})
            metrics.valid_pixels = existing_metrics.get('valid_pixels_per_timestep', {})
            metrics.outliers_removed = existing_metrics.get('outliers_removed_per_timestep', {})
        print(f"Loaded existing metrics from {metrics_path}")
    
    # Get all existing files
    csv_files = sorted(glob.glob(os.path.join(input_dir, "snapshot_*.csv")))
    print(f"Found {len(csv_files)} original snapshot files")
    
    # Read all files into a single DataFrame
    dfs = []
    for f in csv_files:
        # Extract date from filename
        date_str = os.path.basename(f).replace("snapshot_", "").replace(".csv", "")
        date = pd.to_datetime(date_str)
        
        df = pd.read_csv(f)
        df['date'] = date
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(full_df)} rows of data")
    
    # Handle duplicates (same grid_id on same date)
    # This was an issue before, so we keep this safety measure
    full_df = full_df.groupby(['date', 'grid_id']).mean(numeric_only=True).reset_index()
    
    # Pivot to wide format: Index=Date, Columns=GridID, Values=Features
    # We need to interpolate all features
    features = [c for c in full_df.columns if c not in ['date', 'grid_id']]
    print(f"Features to interpolate: {features}")
    
    # Create a complete 5-day date range
    start_date = full_df['date'].min()
    end_date = full_df['date'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='5D')
    print(f"Target date range: {start_date.date()} to {end_date.date()} (Every 5 days)")
    print(f"Total expected steps: {len(date_range)}")
    
    # Get all unique grid_ids from the data
    all_grid_ids = full_df['grid_id'].unique()
    print(f"Total unique grid_ids in data: {len(all_grid_ids)}")
    
    # Use chlorophyll_a to track interpolation (main feature of interest)
    # Pivot creates NaN where a grid_id doesn't have data for a date
    chl_pivot_original = full_df.pivot(index='date', columns='grid_id', values='chlorophyll_a')
    
    # Reindex to the target 5-day date range (creates new rows for missing dates)
    chl_pivot_reindexed = chl_pivot_original.reindex(date_range)
    
    # Interpolate temporally (fills NaN where there's data before/after)
    chl_pivot_interpolated = chl_pivot_reindexed.interpolate(method='time')
    
    # Fill edges with forward-fill and backward-fill
    chl_pivot_filled = chl_pivot_interpolated.ffill().bfill()
    
    # Identify grid_ids that have at least one valid value after interpolation
    # These are the "water pixels" - drop any column that is still all NaN
    valid_grid_ids = chl_pivot_filled.columns[chl_pivot_filled.notna().any()].tolist()
    total_water_pixels = len(valid_grid_ids)
    
    print(f"Grid_ids with at least one valid value: {total_water_pixels}")
    print(f"Grid_ids that are all NaN (removed): {len(all_grid_ids) - total_water_pixels}")
    
    # Filter to only valid grid_ids
    chl_pivot_filled = chl_pivot_filled[valid_grid_ids]
    chl_pivot_reindexed_filtered = chl_pivot_reindexed[valid_grid_ids]
    
    # Set total_water_pixels in metrics (this is FIXED for all timesteps)
    metrics.total_water_pixels = total_water_pixels
    
    # Track original and interpolated pixels per date
    print("\nTracking pixels per date...")
    print(f"{'Date':<12} {'Original':>10} {'Interpolated':>12} {'Total':>10}")
    print("-" * 46)
    
    total_original = 0
    total_interpolated = 0
    
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        
        # Get values before and after interpolation+fill for this date
        before = chl_pivot_reindexed_filtered.loc[date]
        after = chl_pivot_filled.loc[date]
        
        # Original = non-NaN before interpolation
        original_count = before.notna().sum()
        
        # Interpolated = was NaN before but has value now
        was_nan = before.isna()
        has_value_now = after.notna()
        interpolated_mask = was_nan & has_value_now
        interpolated_count = interpolated_mask.sum()
        
        # Final = should equal total_water_pixels
        final_count = after.notna().sum()
        
        # Store in metrics
        metrics.add_original_pixels(date_str, int(original_count))
        metrics.add_interpolated_pixels(date_str, int(interpolated_count))
        
        total_original += original_count
        total_interpolated += interpolated_count
        
        # Verify consistency
        assert final_count == total_water_pixels, f"Inconsistency on {date_str}: {final_count} != {total_water_pixels}"
        
        print(f"{date_str:<12} {original_count:>10} {interpolated_count:>12} {final_count:>10}")
    
    print("-" * 46)
    print(f"{'TOTAL':<12} {total_original:>10} {total_interpolated:>12} {total_water_pixels:>10}")
    print(f"\nAll timesteps have exactly {total_water_pixels} water pixels ✓")

    
    # Process each feature - only for valid grid_ids
    interpolated_dfs = []
    
    for feature in features:
        print(f"Processing feature: {feature}")
        # Pivot
        pivot_df = full_df.pivot(index='date', columns='grid_id', values=feature)
        
        # Filter to only valid grid_ids (same as chlorophyll)
        pivot_df = pivot_df[[c for c in valid_grid_ids if c in pivot_df.columns]]
        
        # Reindex to the 5-day range
        pivot_df = pivot_df.reindex(date_range)
        
        # Interpolate (Time-based linear interpolation)
        pivot_df_interp = pivot_df.interpolate(method='time')
        
        # Fill edges with forward-fill and backward-fill
        pivot_df_interp = pivot_df_interp.ffill().bfill()
        
        # Unstack back to long format
        long_df = pivot_df_interp.unstack().reset_index()
        long_df.columns = ['grid_id', 'date', feature]
        
        if len(interpolated_dfs) == 0:
            interpolated_dfs.append(long_df)
        else:
            # Merge with existing results
            interpolated_dfs[0] = pd.merge(interpolated_dfs[0], long_df, on=['date', 'grid_id'])
            
    final_df = interpolated_dfs[0]
    
    # Save files
    count = 0
    for date in date_range:
        day_data = final_df[final_df['date'] == date].copy()
        
        # Format date column to string YYYY-MM-DD
        day_data['date'] = day_data['date'].dt.strftime('%Y-%m-%d')
        
        # Reorder columns to match original if possible, or at least keep grid_id first
        # Ensure 'date' is included
        cols = ['grid_id', 'date'] + [c for c in day_data.columns if c not in ['grid_id', 'date']]
        day_data_save = day_data[cols]
        
        date_str = date.strftime('%Y-%m-%d')
        filename = f"snapshot_{date_str}.csv"
        output_path = os.path.join(output_dir, filename)
        
        day_data_save.to_csv(output_path, index=False)
        count += 1
    
    # Save updated metrics
    output_metrics_path = os.path.join(output_dir, 'processing_metrics.json')
    metrics.save(output_metrics_path)
    metrics.print_summary()
        
    print(f"Successfully generated {count} files in {output_dir}")

if __name__ == "__main__":
    interpolate_5day_gaps()
