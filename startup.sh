#!/bin/bash

# This script will be executed before the app starts
# It installs the ODBC drivers needed for connecting to SQL Server

# Update package repositories
apt-get update

# Install required dependencies
apt-get install -y curl gnupg2 unixodbc unixodbc-dev

# Add Microsoft repository key
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

# Add Microsoft repository
curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Update repositories again after adding Microsoft repository
apt-get update

# Install ODBC Drivers accepting the EULA
ACCEPT_EULA=Y apt-get install -y msodbcsql17 msodbcsql18

# Verify ODBC installation
odbcinst -j