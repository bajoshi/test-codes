#!/bin/bash 

echo "Starting test bash script at $(date)"
echo ""
echo "This script will call a dummy Python script -- $1"
echo "that saves a numpy array each time it is called."
echo "It expects the name of the Python script as its first argument."

# I know it is redundant to use both i and 
# counter below but I want to test both ways.
# This is a C style for loop for bash. 
# I like it better than any other way.
# Tried to use a range like so {start..end} 
# but that didn't work.
# Note the C style for loop and incrementation
# doesn't require a $ sign to be placed in front 
# of i or counter when it is incremented. Also
# note the usage of double parentheses.
counter=0
total_runs=10
for (( i=0; i <= $total_runs; i++ ))
do
  echo "Running Python test script for the $i $counter time"
  python $1 $counter
  ((counter++))
done

echo ""
echo "Finished running $total_runs times."
