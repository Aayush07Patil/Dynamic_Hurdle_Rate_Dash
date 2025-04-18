#!/bin/bash

# Install ODBC Driver for Azure Web App Linux
# This script will run during startup

# Update package listings
apt-get update

# Install required dependencies
apt-get install -y unixodbc unixodbc-dev gnupg2 curl

# Add Microsoft repository keys
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

# Add Microsoft repository
curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Update package listings again
apt-get update

# Install ODBC Driver 17 for SQL Server (accepting EULA)
ACCEPT_EULA=Y apt-get install -y msodbcsql17

# Verify installation
odbcinst -j