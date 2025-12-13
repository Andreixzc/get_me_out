"""
Grid-Based Chlorophyll-a Time Series Extraction for Trs Marias Reservoir
REFACTORED: Exports one CSV file per day for easier visualization
Author: Time Series Analysis for Water Quality Prediction
"""

import ee
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import sys
import os
from processing_metrics import ProcessingMetrics

# Initialize Earth Engine
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

# Add CHL-CONNECT to path
sys.path.append('./Chl-CONNECT')

class GridChlorophyllExtractor:
    def __init__(self, aoi, grid_points=3000, start_date='2020-01-01', end_date='2023-12-31'):
        """
        Initialize the grid-based chlorophyll extractor
        
        Args:
            aoi: ee.Geometry of the area of interest
            grid_points: Number of grid points to sample
            start_date: Start date for time series
            end_date: End date for time series
        """
        self.aoi = aoi
        self.grid_points = grid_points
        self.start_date = start_date
        self.end_date = end_date
        self.grid_coordinates = None
        self.time_series_data = None
        self.metrics = ProcessingMetrics()
        
    def create_sampling_grid(self):
        """
        Create systematic grid points within the AOI
        """
        # Get AOI bounds
        bounds = self.aoi.bounds().getInfo()['coordinates'][0]
        
        # Extract min/max coordinates
        lons = [point[0] for point in bounds]
        lats = [point[1] for point in bounds]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Calculate grid dimensions (approximately square grid)
        grid_size = int(math.sqrt(self.grid_points))
        
        print(f"Creating {self.grid_points} grid points (~{grid_size}x{grid_size} grid)...")
        actual_points = grid_size * grid_size
        
        print(f"Using {grid_size}x{grid_size} grid = {actual_points} points")
        
        # Create grid coordinates
        lon_step = (max_lon - min_lon) / (grid_size - 1)
        lat_step = (max_lat - min_lat) / (grid_size - 1)
        
        grid_coords = []
        for i in range(grid_size):
            for j in range(grid_size):
                lon = min_lon + j * lon_step
                lat = min_lat + i * lat_step
                grid_coords.append({
                    'id': f'grid_{i}_{j}',
                    'lon': lon,
                    'lat': lat,
                    'geometry': ee.Geometry.Point([lon, lat])
                })
        
        self.grid_coordinates = grid_coords
        self.metrics.total_grid_points = len(grid_coords)
        print(f"Grid created with {len(grid_coords)} points")
        return grid_coords
    
    def mask_clouds_and_water(self, image):
        """
        Apply cloud and water masks to Sentinel-2 image
        """
        # Cloud mask using QA60
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        cloud_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                    qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        
        # More permissive water mask - only exclude obviously non-water areas
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # Basic water mask (less restrictive)
        water_mask = ndwi.gt(-0.1).And(
                    image.select('B8').lt(2000)).And(
                    image.select('B4').lt(2000))
        
        # Combine masks
        final_mask = cloud_mask.And(water_mask)
        
        return image.updateMask(final_mask)
    
    def calculate_rrs(self, image):
        """
        Convert Sentinel-2 surface reflectance to Remote Sensing Reflectance (Rrs)
        """
        # Sentinel-2 bands for water quality (scaled 0-10000)
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A']
        
        # Convert to reflectance (0-1)
        reflectance = image.select(bands).multiply(0.0001)
        
        # Convert to Rrs (reflectance / )
        rrs = reflectance.divide(math.pi)
        
        # Rename bands to match CHL-CONNECT requirements for MSI
        rrs_bands = rrs.select(bands, 
                              ['Rrs443', 'Rrs490', 'Rrs560', 'Rrs665', 
                               'Rrs705', 'Rrs740', 'Rrs783', 'Rrs865'])
        
        return image.addBands(rrs_bands)
    
    def extract_time_series(self):
        """
        Extract time series data for all grid points
        """
        print("Loading Sentinel-2 collection...")
        
        # Load Sentinel-2 collection
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(self.aoi)
                        .filterDate(self.start_date, self.end_date)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                        .map(self.mask_clouds_and_water)
                        .map(self.calculate_rrs))
        
        collection_size = s2_collection.size().getInfo()
        print(f"Found {collection_size} images after filtering")
        
        if collection_size == 0:
            print("No valid images found! Check date range and cloud cover threshold.")
            return None
        
        # Process in batches to avoid memory issues
        batch_size = 100
        all_time_series = []
        
        print(f"Processing {len(self.grid_coordinates)} grid points in batches of {batch_size}...")
        
        for batch_start in range(0, len(self.grid_coordinates), batch_size):
            batch_end = min(batch_start + batch_size, len(self.grid_coordinates))
            batch_points = self.grid_coordinates[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(self.grid_coordinates)-1)//batch_size + 1}")
            print(f"Grid points {batch_start+1} to {batch_end}")
            
            # Create a FeatureCollection of grid points for this batch
            batch_geometries = ee.FeatureCollection([
                ee.Feature(grid_point['geometry'], {
                    'grid_id': grid_point['id'],
                    'lon': grid_point['lon'],
                    'lat': grid_point['lat']
                }) for grid_point in batch_points
            ])
            
            try:
                # Extract time series for all points in this batch
                def extract_batch_values(image):
                    # Extract Rrs values at all batch points
                    rrs_bands = ['Rrs443', 'Rrs490', 'Rrs560', 'Rrs665', 
                                'Rrs705', 'Rrs740', 'Rrs783', 'Rrs865']
                    
                    # Sample at all grid points in batch
                    samples = image.select(rrs_bands).sampleRegions(
                        collection=batch_geometries,
                        scale=20,
                        tileScale=2
                    )
                    
                    # Add image metadata to each sample
                    def add_metadata(feature):
                        return feature.set({
                            'date': image.date().format('YYYY-MM-dd'),
                            'system:time_start': image.get('system:time_start'),
                            'scene_id': image.get('PRODUCT_ID')
                        })
                    
                    return samples.map(add_metadata)
                
                # Map over collection for this batch
                batch_time_series = s2_collection.map(extract_batch_values).flatten()
                
                # Convert to list and add to results
                batch_data = batch_time_series.getInfo()
                
                # Process features
                for feature in batch_data['features']:
                    props = feature['properties']
                    # Only add if we have valid Rrs values
                    if props.get('Rrs665') is not None:
                        all_time_series.append(props)
                        
                print(f"Batch completed. Found {len([f for f in batch_data['features'] if f['properties'].get('Rrs665') is not None])} valid observations")
                        
            except Exception as e:
                print(f"Error processing batch {batch_start//batch_size + 1}: {e}")
                continue
        
        # Convert to DataFrame
        if all_time_series:
            df = pd.DataFrame(all_time_series)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['date', 'grid_id']).reset_index(drop=True)
            
            self.time_series_data = df
            
            # Track metrics: water_points and valid_pixels per date
            for date in df['date'].dt.date.unique():
                date_str = str(date)
                date_data = df[df['date'].dt.date == date]
                # water_points = number of grid points that returned data (were detected as water)
                water_count = len(date_data)
                self.metrics.add_water_points(date_str, water_count)
                # valid_pixels = pixels with valid Rrs values (these are all valid since we filtered earlier)
                self.metrics.add_valid_pixels(date_str, water_count)
            
            print(f"\nExtracted {len(df)} valid observations across {df['grid_id'].nunique()} grid points")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Unique dates: {df['date'].dt.date.nunique()}")
            
            return df
        else:
            print("No valid time series data extracted!")
            return None
    
    def calculate_chlorophyll(self):
        """
        Calculate chlorophyll-a using CHL-CONNECT for each time series point
        """
        if self.time_series_data is None:
            print("No time series data available. Run extract_time_series() first.")
            return None
        
        print("\nCalculating chlorophyll-a concentrations...")
        
        try:
            from common.Chl_CONNECT import Chl_CONNECT
        except ImportError:
            print("CHL-CONNECT not available. Please install the library.")
            return None
        
        # Add chlorophyll columns
        chl_values = []
        class_values = []
        outlier_flags = []  # Track outliers for metrics
        
        total_rows = len(self.time_series_data)
        
        for idx, row in self.time_series_data.iterrows():
            if idx % 100 == 0:
                print(f"Processing {idx}/{total_rows}...")
            
            try:
                # Prepare Rrs input for CHL-CONNECT (MSI sensor - only 5 bands needed)
                rrs_values = [
                    row['Rrs443'], row['Rrs490'], row['Rrs560'], row['Rrs665'], row['Rrs705']
                ]
                
                # Check for valid values
                if any(pd.isna(rrs_values)):
                    chl_values.append(np.nan)
                    class_values.append(np.nan)
                    outlier_flags.append(False)  # Not an outlier, just missing data
                    continue
                
                # Convert to numpy array with correct shape (1, 5) for single pixel
                rrs_input = np.array(rrs_values).reshape(1, -1)
                
                # Initialize CHL-CONNECT
                chl_conn = Chl_CONNECT(Rrs_input=rrs_input, sensor='MSI')
                
                # Get results
                chl_concentration = chl_conn.Chl_comb[0] if hasattr(chl_conn.Chl_comb, '__len__') else chl_conn.Chl_comb
                water_class = chl_conn.Class[0] if hasattr(chl_conn.Class, '__len__') else chl_conn.Class
                
                # Filter invalid values (outliers and negative values)
                if chl_concentration > 100 or chl_concentration < 0:
                    chl_values.append(np.nan)
                    class_values.append(np.nan)
                    outlier_flags.append(True)  # This is an outlier
                else:
                    chl_values.append(chl_concentration)
                    class_values.append(water_class)
                    outlier_flags.append(False)
                
            except Exception as e:
                print(f"Error calculating chlorophyll for row {idx}: {e}")
                chl_values.append(np.nan)
                class_values.append(np.nan)
                outlier_flags.append(False)  # Error, not an outlier
        
        # Add to dataframe
        self.time_series_data['chlorophyll_a'] = chl_values
        self.time_series_data['water_class'] = class_values
        self.time_series_data['_is_outlier'] = outlier_flags
        self.time_series_data['log_chl_a'] = np.log10(
            np.maximum(self.time_series_data['chlorophyll_a'], 0.01))
        
        # Track outliers per date for metrics
        for date in self.time_series_data['date'].dt.date.unique():
            date_str = str(date)
            date_data = self.time_series_data[self.time_series_data['date'].dt.date == date]
            outlier_count = date_data['_is_outlier'].sum()
            if outlier_count > 0:
                self.metrics.add_outliers_removed(date_str, int(outlier_count))
        
        # Remove temporary column
        self.time_series_data = self.time_series_data.drop(columns=['_is_outlier'])
        
        valid_count = pd.notna(chl_values).sum()
        print(f"\n Calculated chlorophyll for {valid_count}/{total_rows} observations")
        
        return self.time_series_data
    
    def export_results(self, output_dir='daily_snapshots'):
        """
        Export results as one CSV file per day
        
        Args:
            output_dir: Directory to save daily CSV files
        """
        if self.time_series_data is None:
            print("No data to export!")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Group by date
        dates = self.time_series_data['date'].dt.date.unique()
        dates = sorted(dates)
        
        print(f"\n{'='*60}")
        print(f"Exporting {len(dates)} daily snapshots to '{output_dir}/'")
        print(f"{'='*60}")
        
        exported_files = []
        
        for date in dates:
            # Filter data for this date
            daily_data = self.time_series_data[
                self.time_series_data['date'].dt.date == date
            ].copy()
            
            # Create filename with date
            date_str = date.strftime('%Y-%m-%d')
            filename = f"{output_dir}/snapshot_{date_str}.csv"
            
            # Export
            daily_data.to_csv(filename, index=False)
            exported_files.append(filename)
            
            # Print summary for this day
            n_points = len(daily_data)
            if 'chlorophyll_a' in daily_data.columns:
                valid_chl = daily_data['chlorophyll_a'].dropna()
                if len(valid_chl) > 0:
                    print(f"   {date_str}: {n_points:4d} points | "
                          f"Chl-a: ={valid_chl.mean():5.2f} mg/m "
                          f"[{valid_chl.min():5.2f}, {valid_chl.max():5.2f}]")
                else:
                    print(f"   {date_str}: {n_points:4d} points (no valid chlorophyll)")
            else:
                print(f"   {date_str}: {n_points:4d} points")
        
        # Print overall summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total dates: {len(dates)}")
        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"Total observations: {len(self.time_series_data)}")
        print(f"Average points per date: {len(self.time_series_data)/len(dates):.1f}")
        print(f"Files exported to: {output_dir}/")
        
        if 'chlorophyll_a' in self.time_series_data.columns:
            valid_chl = self.time_series_data['chlorophyll_a'].dropna()
            print(f"\nOverall chlorophyll statistics:")
            print(f"  Valid observations: {len(valid_chl)}")
            print(f"  Range: {valid_chl.min():.3f} - {valid_chl.max():.3f} mg/m")
            print(f"  Mean: {valid_chl.mean():.3f} mg/m")
            print(f"  Median: {valid_chl.median():.3f} mg/m")
            print(f"  Std Dev: {valid_chl.std():.3f} mg/m")
        
        print(f"{'='*60}\n")
        
        # Save processing metrics to JSON
        metrics_path = os.path.join(output_dir, 'processing_metrics.json')
        self.metrics.save(metrics_path)
        self.metrics.print_summary()
        
        return exported_files

def main():
    """
    Main execution function
    """
    # Load AOI from Area.json
    try:
        import json
        with open('Area.json', 'r') as f:
            aoi_data = json.load(f)
        # Get first feature's geometry
        coords = aoi_data['features'][0]['geometry']['coordinates']
        aoi = ee.Geometry.Polygon(coords)
        print(" Loaded AOI from Area.json")
    except Exception as e:
        print(f" Error loading Area.json: {e}")
        # Fallback to hardcoded coordinates
        aoi = ee.Geometry.Polygon([[[-45.559114, -18.954365], [-45.559114, -18.212409], 
                                   [-44.839706, -18.212409], [-44.839706, -18.954365], 
                                   [-45.559114, -18.954365]]])
        print(" Using fallback AOI coordinates")
    
    # Parameters
    extractor = GridChlorophyllExtractor(
        aoi=aoi,
        grid_points=50000,  # 100x100 grid
        start_date='2024-06-01',
        end_date='2024-11-30'
    )
    
    print("\n" + "="*60)
    print("Grid-Based Chlorophyll Extraction - Daily Snapshots")
    print("="*60)
    print("Area: Trs Marias Reservoir, Brazil")
    print(f"Grid: {extractor.grid_points} points")
    print(f"Period: {extractor.start_date} to {extractor.end_date}")
    print("="*60 + "\n")
    
    # Step 1: Create sampling grid
    grid_coords = extractor.create_sampling_grid()
    
    # Step 2: Extract time series
    time_series_df = extractor.extract_time_series()
    
    if time_series_df is not None:
        # Step 3: Calculate chlorophyll (optional - requires CHL-CONNECT)
        try:
            chlorophyll_df = extractor.calculate_chlorophyll()
        except Exception as e:
            print(f" Chlorophyll calculation skipped: {e}")
            chlorophyll_df = time_series_df
        
        # Step 4: Export results (one file per day)
        exported_files = extractor.export_results('daily_snapshots')
        
        print(f"\n Processing complete! Exported {len(exported_files)} daily files.")
    
        return extractor
    
    else:
        print(" Failed to extract time series data.")
        return None

# Example usage
if __name__ == "__main__":
    results = main()