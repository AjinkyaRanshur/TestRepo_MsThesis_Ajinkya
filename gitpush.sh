#!/bin/bash
###########################

echo What message do you want to put?
read message

#########################

# add all added/modified files
git add -A
# commit changes
git commit -m "$message"
# push to git remote repository
git push origin main
###########################
echo Press Enter...
read

