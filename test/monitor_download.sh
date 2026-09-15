#!/bin/bash

FILE="the-fake-or-real-dataset.zip"
INTERVAL=5  # Check every 5 seconds

echo "Starting download monitoring..."
echo "Initial size: $(ls -lh "$FILE" 2>/dev/null | awk '{print $5}')"

PREV_SIZE=$(stat -f%z "$FILE" 2>/dev/null || echo 0)
sleep $INTERVAL

for i in {1..60}; do
    if [ -f "$FILE" ]; then
        CURR_SIZE=$(stat -f%z "$FILE")
        DIFF=$((CURR_SIZE - PREV_SIZE))
        SPEED=$((DIFF / INTERVAL))
        
        # Convert to MB
        SPEED_MB=$(echo "scale=2; $SPEED / 1048576" | bc)
        CURR_MB=$(echo "scale=2; $CURR_SIZE / 1048576" | bc)
        
        echo "[$i] Current: ${CURR_MB} MB | Speed: ${SPEED_MB} MB/s"
        
        PREV_SIZE=$CURR_SIZE
    else
        echo "File not found, waiting..."
    fi
    sleep $INTERVAL
done
