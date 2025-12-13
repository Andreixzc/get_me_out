"""
Processing Metrics Module for Grid-Based Chlorophyll Extraction
Tracks and exports metrics about data processing pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import json
from datetime import datetime


@dataclass
class ProcessingMetrics:
    """
    Stores metrics about the data processing pipeline
    
    Attributes:
        total_grid_points: Total number of uniform points in the sampling grid
        total_water_pixels: Fixed number of water pixels (grid_ids with at least one valid observation)
        original_pixels: Dict[date_str, count] - Original valid pixels per timestep (before interpolation)
        interpolated_pixels: Dict[date_str, count] - Pixels that were interpolated per timestep
        outliers_removed: Dict[date_str, count] - Pixels removed as outliers per timestep
    """
    total_grid_points: int = 0
    total_water_pixels: int = 0  # Fixed: grid_ids with at least one valid chlorophyll value
    original_pixels: Dict[str, int] = field(default_factory=dict)  # Renamed from valid_pixels
    interpolated_pixels: Dict[str, int] = field(default_factory=dict)
    outliers_removed: Dict[str, int] = field(default_factory=dict)
    
    # Legacy fields for backwards compatibility
    water_points: Dict[str, int] = field(default_factory=dict)
    valid_pixels: Dict[str, int] = field(default_factory=dict)
    
    def add_water_points(self, date: str, count: int):
        """Record number of water points for a given date (legacy)"""
        self.water_points[date] = count
    
    def add_valid_pixels(self, date: str, count: int):
        """Record number of valid pixels for a given date (legacy)"""
        self.valid_pixels[date] = count
    
    def add_original_pixels(self, date: str, count: int):
        """Record number of original pixels (before interpolation) for a given date"""
        self.original_pixels[date] = count
    
    def add_outliers_removed(self, date: str, count: int):
        """Record number of outliers removed for a given date"""
        if date in self.outliers_removed:
            self.outliers_removed[date] += count
        else:
            self.outliers_removed[date] = count
    
    def add_interpolated_pixels(self, date: str, count: int):
        """Record number of interpolated pixels for a given date"""
        self.interpolated_pixels[date] = count
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary for JSON export"""
        # Calculate per-timestep breakdown with consistency check
        timestep_breakdown = {}
        for date in sorted(set(self.original_pixels.keys()) | set(self.interpolated_pixels.keys())):
            orig = self.original_pixels.get(date, 0)
            interp = self.interpolated_pixels.get(date, 0)
            total = orig + interp
            timestep_breakdown[date] = {
                "original": orig,
                "interpolated": interp,
                "total": total
            }
        
        return {
            "total_grid_points": self.total_grid_points,
            "total_water_pixels": self.total_water_pixels,
            "timesteps": timestep_breakdown,
            "summary": {
                "total_timesteps": len(timestep_breakdown),
                "total_water_pixels": self.total_water_pixels,
                "total_original_observations": sum(self.original_pixels.values()),
                "total_interpolated": sum(self.interpolated_pixels.values()),
                "total_outliers_removed": sum(self.outliers_removed.values()),
            },
            # Legacy fields
            "legacy": {
                "water_points_per_timestep": self.water_points,
                "valid_pixels_per_timestep": self.valid_pixels,
                "outliers_removed_per_timestep": self.outliers_removed,
            }
        }
    
    def save(self, filepath: str):
        """Save metrics to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {filepath}")
    
    def print_summary(self):
        """Print a summary of the metrics"""
        print("\n" + "="*60)
        print("PROCESSING METRICS SUMMARY")
        print("="*60)
        print(f"Total grid points (sampling): {self.total_grid_points}")
        print(f"Total water pixels (fixed):   {self.total_water_pixels}")
        print(f"Total timesteps:              {len(self.original_pixels)}")
        print("-"*60)
        print(f"Total original observations:  {sum(self.original_pixels.values())}")
        print(f"Total interpolated pixels:    {sum(self.interpolated_pixels.values())}")
        print(f"Total outliers removed:       {sum(self.outliers_removed.values())}")
        print("="*60 + "\n")
