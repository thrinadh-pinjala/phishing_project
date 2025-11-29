#!/bin/bash
# GitHub Repository Setup Script
# This script configures your GitHub repository with description, topics, and features

REPO_OWNER="thrinadh-pinjala"
REPO_NAME="phishing_project"
REPO_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}"

echo "========================================="
echo "GitHub Repository Configuration"
echo "========================================="
echo "Repository: $REPO_URL"
echo ""

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

echo "✅ GitHub CLI found"
echo ""

# Authenticate if needed
echo "Checking authentication..."
gh auth status

echo ""
echo "Setting repository description..."
gh repo edit ${REPO_OWNER}/${REPO_NAME} \
    --description "Comprehensive ML Pipeline for Phishing Detection using ensemble learning, correlation analysis, and interpretability frameworks"

echo ""
echo "Adding repository topics..."
gh repo edit ${REPO_OWNER}/${REPO_NAME} \
    --add-topic "phishing-detection" \
    --add-topic "machine-learning" \
    --add-topic "cybersecurity" \
    --add-topic "classification" \
    --add-topic "data-science" \
    --add-topic "python" \
    --add-topic "sklearn"

echo ""
echo "Enabling discussions..."
# Note: This requires authenticated access
gh repo edit ${REPO_OWNER}/${REPO_NAME} --enable-discussions

echo ""
echo "✅ Repository configuration complete!"
echo ""
echo "Repository: $REPO_URL"
echo "Description: Comprehensive ML Pipeline for Phishing Detection"
echo "Topics: phishing-detection, machine-learning, cybersecurity, classification, data-science, python, sklearn"
echo "Discussions: Enabled"
