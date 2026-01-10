#!/bin/bash

# Helper script to format and label SSD as "KmiDi-DONE"
# WARNING: This will erase all data on the selected disk!

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SSD_LABEL="KmiDi-DONE"
SSD_MOUNT="/Volumes/${SSD_LABEL}"

echo -e "${BLUE}SSD Preparation Script for KmiDi-DONE${NC}"
echo ""

# Check if already mounted with correct label
if [ -d "$SSD_MOUNT" ]; then
    echo -e "${GREEN}SSD already mounted at ${SSD_MOUNT}${NC}"
    exit 0
fi

# List external disks
echo -e "${YELLOW}Available external disks:${NC}"
diskutil list | grep -A 5 "external, physical" || echo "No external disks found"

echo ""
echo -e "${YELLOW}Please identify the disk identifier (e.g., disk4) for your SSD${NC}"
echo -e "${RED}WARNING: Formatting will erase all data on the selected disk!${NC}"
echo ""
read -p "Enter disk identifier (e.g., disk4) or 'cancel' to abort: " DISK_ID

if [ "$DISK_ID" = "cancel" ] || [ -z "$DISK_ID" ]; then
    echo "Aborted."
    exit 1
fi

# Verify disk exists
if ! diskutil list | grep -q "^/dev/$DISK_ID"; then
    echo -e "${RED}Error: Disk $DISK_ID not found${NC}"
    exit 1
fi

# Show disk info
echo ""
echo -e "${BLUE}Disk information:${NC}"
diskutil info "/dev/$DISK_ID" | grep -E "Device Node|Disk Size|Volume Name"

echo ""
read -p "Format this disk as '${SSD_LABEL}'? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Format the disk
echo ""
echo -e "${BLUE}Formatting disk as ${SSD_LABEL}...${NC}"
diskutil eraseDisk APFS "${SSD_LABEL}" "/dev/$DISK_ID"

# Wait for mount
sleep 2

# Verify mount
if [ -d "$SSD_MOUNT" ]; then
    echo -e "${GREEN}SSD successfully formatted and mounted at ${SSD_MOUNT}${NC}"
else
    echo -e "${YELLOW}SSD formatted but not yet mounted. Please wait a moment and check /Volumes/${NC}"
fi
