import pandas as pd

# Load data
df = pd.read_csv(r'C:\Users\DELL-2025\Desktop\internship\Task_2\Unemployment in India.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Print columns for debugging
print("Columns:", df.columns)

# Fix date column (find correct name even if spacing or different name)
date_col = [col for col in df.columns if 'date' in col.lower()][0]

# Strip leading/trailing spaces in date values
df[date_col] = df[date_col].astype(str).str.strip()

# Convert date to datetime
df[date_col] = pd.to_datetime(df[date_col], format='%d-%m-%Y')

# Rename columns for simplicity
df = df.rename(columns={
    'Estimated Unemployment Rate (%)': 'UnemploymentRate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'LabourParticipationRate'
})

# Convert numeric columns to proper types
df['UnemploymentRate'] = pd.to_numeric(df['UnemploymentRate'], errors='coerce')
df['Employed'] = pd.to_numeric(df['Employed'], errors='coerce')
df['LabourParticipationRate'] = pd.to_numeric(df['LabourParticipationRate'], errors='coerce')

# Remove duplicates
df = df.drop_duplicates()

# Check missing values
print(df.isnull().sum())

#______________________________________________________________________________________

# Data Analysis

#______________________________________________________________________________________


# 1. Basic info
print(df.info())
print(df.describe())

# 2. Unemployment rate overview
print("Average Unemployment:", df['UnemploymentRate'].mean())
print("Highest Unemployment:", df['UnemploymentRate'].max())
print("Lowest Unemployment:", df['UnemploymentRate'].min())

# Region with highest unemployment
region_max = df.loc[df['UnemploymentRate'].idxmax()]
print("Region with highest unemployment:", region_max['Region'])

# Region with lowest unemployment
region_min = df.loc[df['UnemploymentRate'].idxmin()]
print("Region with lowest unemployment:", region_min['Region'])

# 3. Trend over time (monthly)
monthly_avg = df.groupby(df['Date'].dt.to_period('M'))['UnemploymentRate'].mean()
print(monthly_avg)

# 4. Region-wise average unemployment
region_avg = df.groupby('Region')['UnemploymentRate'].mean()
print(region_avg)

# 5. Area-wise comparison
area_avg = df.groupby('Area')['UnemploymentRate'].mean()
print(area_avg)
