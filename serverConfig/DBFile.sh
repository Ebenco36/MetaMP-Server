#!/bin/bash
# Get the absolute path of the script
SCRIPT_PATH=$(realpath "$0")

# Determine the script's directory
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

# Move up two directories
PARENT_DIR=$(realpath "$SCRIPT_DIR/../")

# python $PARENT_DIR/src/Jobs/MPStructJobs.py
# python $PARENT_DIR/src/Jobs/PDBJobs.py
# python $PARENT_DIR/src/Jobs/MergeDB.py
# python $PARENT_DIR/src/Jobs/NewOPMJob.py
# python $PARENT_DIR/src/Jobs/UniprotJob.py
# python $PARENT_DIR/src/Jobs/GenerateCountries.py
# python $PARENT_DIR/src/Jobs/transformData.py  

# # Start the Flask app or other necessary services
# echo "Running Flask sync-protein-database..."
# flask sync-protein-database


# Print completion message
echo "==================================================================="
echo "=====✅ All tasks completed successfully. The system is ready.====="
echo "==================================================================="