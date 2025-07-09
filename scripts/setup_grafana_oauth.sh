#!/bin/bash

# Setup script for Grafana Google OAuth and Gmail SMTP

echo "🔐 Grafana Google OAuth & Gmail SMTP Setup"
echo "========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp env.example .env
fi

# Prompt for Google OAuth credentials
echo "📝 Google OAuth Setup"
echo "-------------------"
echo "1. Go to https://console.cloud.google.com/"
echo "2. Create a new project or select existing"
echo "3. Enable Google+ API"
echo "4. Create OAuth 2.0 Client ID"
echo "5. Set authorized redirect URI: http://localhost:3000/login/google"
echo ""
read -p "Enter Google OAuth Client ID: " client_id
read -p "Enter Google OAuth Client Secret: " client_secret

# Prompt for Gmail app password
echo ""
echo "📧 Gmail SMTP Setup"
echo "----------------"
echo "1. Go to your Google Account settings"
echo "2. Security → 2-Step Verification → App passwords"
echo "3. Generate new app password for 'Grafana'"
echo ""
read -p "Enter Gmail App Password: " gmail_pass

# Update .env file
sed -i.bak "s/GRAFANA_GOOGLE_CLIENT_ID=.*/GRAFANA_GOOGLE_CLIENT_ID=$client_id/" .env
sed -i.bak "s/GRAFANA_GOOGLE_CLIENT_SECRET=.*/GRAFANA_GOOGLE_CLIENT_SECRET=$client_secret/" .env
sed -i.bak "s/GMAIL_APP_PASSWORD=.*/GMAIL_APP_PASSWORD=$gmail_pass/" .env
rm .env.bak

echo ""
echo "✅ Credentials updated in .env"
echo ""
echo "🔄 Reloading Grafana configuration..."
make grafana-reload

echo ""
echo "🎉 Setup complete! You can now:"
echo "1. Login to Grafana with your Google account"
echo "2. Configure email alerts using Gmail SMTP"