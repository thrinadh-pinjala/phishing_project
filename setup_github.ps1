# GitHub Repository Setup Script (PowerShell)
# This script configures your GitHub repository with description, topics, and features

$REPO_OWNER = "thrinadh-pinjala"
$REPO_NAME = "phishing_project"
$REPO_URL = "https://github.com/$REPO_OWNER/$REPO_NAME"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "GitHub Repository Configuration" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Repository: $REPO_URL" -ForegroundColor Yellow
Write-Host ""

# Check if GitHub CLI is installed
try {
    $ghVersion = gh --version
    Write-Host "✅ GitHub CLI found: $ghVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ GitHub CLI (gh) is not installed." -ForegroundColor Red
    Write-Host "Install it from: https://cli.github.com/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Checking authentication..." -ForegroundColor Cyan
gh auth status

Write-Host ""
Write-Host "Setting repository description..." -ForegroundColor Cyan
& gh repo edit "$REPO_OWNER/$REPO_NAME" `
    --description "Comprehensive ML Pipeline for Phishing Detection using ensemble learning, correlation analysis, and interpretability frameworks"

Write-Host ""
Write-Host "Adding repository topics..." -ForegroundColor Cyan
& gh repo edit "$REPO_OWNER/$REPO_NAME" `
    --add-topic "phishing-detection" `
    --add-topic "machine-learning" `
    --add-topic "cybersecurity" `
    --add-topic "classification" `
    --add-topic "data-science" `
    --add-topic "python" `
    --add-topic "sklearn"

Write-Host ""
Write-Host "Enabling discussions..." -ForegroundColor Cyan
& gh repo edit "$REPO_OWNER/$REPO_NAME" --enable-discussions

Write-Host ""
Write-Host "✅ Repository configuration complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Repository: $REPO_URL" -ForegroundColor Yellow
Write-Host "Description: Comprehensive ML Pipeline for Phishing Detection" -ForegroundColor Yellow
Write-Host "Topics: phishing-detection, machine-learning, cybersecurity, classification, data-science, python, sklearn" -ForegroundColor Yellow
Write-Host "Discussions: Enabled" -ForegroundColor Yellow
