# Loop through each directory in the current directory
for dir in */; do
    # Extract the directory name
    dirname=$(basename "$dir")
    
    # Print the directory name
    echo "Running main.py in folder: $dirname"
    
    # Navigate into the directory
    cd "$dir" || continue
    
    # Check if main.py exists in the directory
    if [ -f "main.py" ]; then
        # Run python main.py
        python main.py >log.txt
    fi
    
    # Return to the parent directory
    cd ..
done
