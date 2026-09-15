#!/usr/bin/env python3
import os
import time
from datetime import timedelta

FILE = "the-fake-or-real-dataset.zip"
TOTAL_GB = 16.0
TOTAL_BYTES = int(TOTAL_GB * 1024 * 1024 * 1024)

print("=" * 60)
print("KAGGLE DATASET DOWNLOAD MONITOR")
print("=" * 60)
print(f"Dataset: The Fake or Real Dataset")
print(f"Total Size: {TOTAL_GB} GB")
print("=" * 60)
print()

if not os.path.exists(FILE):
    print(f"Error: {FILE} not found!")
    exit(1)

# Get initial size
prev_size = os.path.getsize(FILE)
prev_time = time.time()

print("Calculating download speed (15 second sample)...")
time.sleep(15)

# Get new size
curr_size = os.path.getsize(FILE)
curr_time = time.time()

# Calculate speed
time_diff = curr_time - prev_time
size_diff = curr_size - prev_size
speed_bps = size_diff / time_diff
speed_mbps = speed_bps / (1024 * 1024)

# Calculate progress
progress_pct = (curr_size / TOTAL_BYTES) * 100
curr_gb = curr_size / (1024 * 1024 * 1024)
curr_mb = curr_size / (1024 * 1024)

# Calculate remaining time
remaining_bytes = TOTAL_BYTES - curr_size
if speed_bps > 0:
    remaining_seconds = int(remaining_bytes / speed_bps)
    remaining_time = timedelta(seconds=remaining_seconds)
else:
    remaining_time = "Unknown"

print()
print("📊 CURRENT STATUS:")
print(f"   Downloaded: {curr_mb:.0f} MB ({curr_gb:.2f} GB)")
print(f"   Progress: {progress_pct:.2f}%")
print()
print("⚡ DOWNLOAD SPEED:")
print(f"   {speed_mbps:.2f} MB/s")
print()
print("⏱️  ESTIMATED TIME REMAINING:")
print(f"   {remaining_time}")
print()
print("=" * 60)

# Visual progress bar
bar_length = 50
filled = int(bar_length * progress_pct / 100)
bar = "█" * filled + "░" * (bar_length - filled)
print(f"[{bar}] {progress_pct:.1f}%")
print("=" * 60)
