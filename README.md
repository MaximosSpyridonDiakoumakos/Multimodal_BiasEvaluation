# MultiModal Bias Evaluation

## Overview
This project's purpose is to test proposed bias evaluation metrics for multimodal machine learning models.
Below are the links for the tested models as found in HuggingFace

This is the bathcelor thesis.

## Basic git commands
Always use git on start.

Receive changes cammnds:
* pull: Fetch all updates from the online source

Push changes
* add . : (Stage) Use add to add in the pipleine the desired changes. Use '.' from root to add all changes
* commit -m "Message": Put a message to the commit
* push: To push changes in the github

Branches
Branches are used for versioning control.
* git checkout -b branchName: Create a new local branch with name branchName
* git checkout branchName: Changes to a branch named branchName
* git branch -d branchName: Delets a branch
* git merge branchName: Merges the branchName to current branch
* git branch: Shows all active branches locally
* git status: Shows the status of current branch





## Script Resources

Text to Image:
stabel-diffusion-v1-5:
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
stable-diffusion-xl-base-1.0 (SDXL)
https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
stable-diffusion-v1-4 (classic baseline)
https://huggingface.co/CompVis/stable-diffusion-v1-4

Image to Text:
blip-image-captioning-base
https://huggingface.co/Salesforce/blip-image-captioning-base
blip-image-captioning-large
https://huggingface.co/Salesforce/blip-image-captioning-large
