# Google-app-Final-Project
App Rating Prediction model
https://colab.research.google.com/drive/1winGVLCXO4VixbU2AWj_OS4c5sFMXW_Q?usp=sharing

# **App Rating Prediction**

from google.colab import drive
drive.mount('/content/drive')
# imports the drive from the google.colab library.

**1. Load the data file using pandas.**

import numpy as np
import pandas as pd

file_path = '/content/drive/MyDrive/googleplaystore.csv'

df = pd.read_csv(file_path)
print("Display Data Frame Head:\n",df.head(),"\n","\n") # Print the first 5 rows of your DataFrame
print("Display summary statistics of the numeric columns:\n",df.describe(),"\n","\n") # Display summary statistics
print("Display Data Frame Info:")
print(df.info()) # Display basic information about the dataset

# **Data Cleaning**

**2. Check for null values in the data. Get the number of null values for each column.**

# Check for null values and get the number of null values for each column
Null_values = df.isnull().sum()
# Display the results
print("Number of null values for each column:\n",Null_values)

**3. Drop records with nulls in any of the columns.**

df.dropna(inplace=True) # Drop rows with any missing values
# Handling Missing Values: Use the dropna method to remove rows with any missing values.
# The inplace=True parameter modifies the original dataset rather than creating a new one.
print("Data Frame Info:")
print(df.info())

duplicates_indices = df[df.duplicated()].groupby(df.columns.tolist()).apply(lambda x: x.index.tolist()).tolist()
print("Indices of duplicate rows:", duplicates_indices)

df.drop_duplicates(inplace=True) # Remove Duplicates
print("Is Null=\n",Null_values)
# Check for null values after removing them
null_values_after_removal = df.isnull().sum()

# Display the results
print("\nNumber of null values for each column after removing:\n",null_values_after_removal)
print("\nDisplay Data Frame Info:")
print(df.info())

**4. Variables seem to have incorrect type and inconsistent formatting. Need to fix them:**

df['Rating'] = df['Rating'].astype(str)

df['Type'] = df['Type'].astype(str)

print("\nDisplay Data Frame Info:") # Rating and Type showing object because of the 'Varies with device' still present in them.
print(df.info())

**a. Size column has sizes in Kb as well as Mb. To analyze, we need to convert these to numeric**
- Extract the numeric value from the column
- Multiply the value by 1,000, if size is mentioned in Mb

def convert_Size(Size):

    if isinstance(Size, str) and 'M' in Size:
        return float(Size.replace('M', ''))*1000
    elif isinstance(Size, str) and 'k' in Size:
        return float(Size.replace('k', ''))
    else:
        return Size

df['Size'] = df['Size'].apply(convert_Size)

**b. Reviews is a numeric field that is loaded as a string field. Convert it to numeric (int/float).**

# Converting 'Reviews' to numeric
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

**c. Installs and price field is currently stored as Float changing it to Int.**
- Treat 1,000,000+ as 1,000,000
- remove ‘+’, ‘,’ from the field, convert it to integer

df['Installs'] = df['Installs'].astype(str)

df['Installs'] = df['Installs'].str.replace('[^\d]', '', regex=True).astype(float) # The reason regex is used here is to clean the 'Installs' column by removing any non-digit characters.



df['Installs'] = df['Installs'].astype(int)

**d. Price field is a string and has $ symbol, removing sign, and converting it to numeric**

df['Price'] = df['Price'].astype(str)

df['Price'] = df['Price'].str.replace('[$]', '', regex=True).astype(float)

df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

df.to_csv('GoogleProject.csv', index=False)
print(df.info())

**Removing textual data.**

# Removing rows that has Varies with device data
value_to_remove = 'Varies with device'

# Initialize a list to store rows to be dropped
rows_to_drop = []

# Loop over rows and check if the value is present in any column
for index, row in df.iterrows():
    if value_to_remove in row.values:
        rows_to_drop.append(index)

# Drop the identified rows
df = df.drop(rows_to_drop)

# Reset the index to make it contiguous
df = df.reset_index(drop=True)

Create a dictionary mapping 'Type' values to index numbers

type_mapping = {'Free': 0, 'Paid': 1}

# Replace 'Type' column with index numbers using the dictionary
df['Type'] = df['Type'].replace(type_mapping)

df.to_csv('GoogleProject.csv', index=False)
print("\nDisplay Data Frame Info:")
print(df.info())

**5. Sanity checks:**
- Average rating should be between 1 and 5 as only these values are allowed on the play store. Droping the rows that have a value outside this range.
- Reviews shouldnt exceeed installs.
- Price of a free application shouldn't exceed 0.

# Convert 'Rating' column to numeric, handling errors with coerce to replace non-numeric values with NaN
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Drop rows with ratings outside the range [1, 5]
df = df[(df['Rating'] >= 1) & (df['Rating'] <= 5)]

# Convert 'Reviews' and 'Installs' columns to numeric (in case they are not already)
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

# Drop rows where reviews are more than installs
df = df[df['Reviews'] <= df['Installs']]

# For free apps, drop rows where the price is greater than 0
df = df[~((df['Type'] == 0) & (df['Price'] > 0))]

df.to_csv('GoogleProject.csv', index=False)

# Droping rows with ratings outside the range [1, 5]
df = df[(df['Rating'] >= 1) & (df['Rating'] <= 5)]

# Droping rows where reviews are more than installs
df = df[df['Reviews'] <= df['Installs']]

# Sanity check For free apps, droping rows where the price is greater than 0
df = df[~((df['Type'] == 'Free') & (df['Price'] > 0))]

df.to_csv('GoogleProject.csv', index=False)
print("\nDisplay Data Frame Info:")
print(df.info())

# **5. Performing Univariate Analysis:**

pip install seaborn


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

**Boxplot for Price**
- Are there any outliers? Think about the price of usual apps on Play Store.

sns.boxplot(x=df['Price'])
plt.title('Box Plot for Price')
plt.xlabel('Price')
plt.show()

max_price_app = df.loc[df['Price'].idxmax(), 'App']
max_price_value = df['Price'].max()

print(f"The app with the highest price is: {max_price_app}")
print(f"The price of the app is: {max_price_value}")

# We can see some outliers.

**Boxplot for Review**
- Are there any apps with very high number of reviews?

max_reviews = df['Reviews'].max()
app_with_max_reviews = df[df['Reviews'] == max_reviews]['App'].values[0]

print(f"The app with the highest number of reviews is '{app_with_max_reviews}' with {max_reviews} reviews.")

# Assuming your DataFrame is named df
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.boxplot(df['Reviews'], vert=False)
plt.title('Box Plot for Reviews')
plt.xlabel('Number of Reviews')
plt.show()

# Yes their are apps with very high review.

**Histogram for Rate**
- How are the ratings distributed?

plt.hist(df['Rating'], bins=20, edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Ratings are mostly between 3 to 5.

**Histogram for Size**
- Note down your observations for the plots made above. Which of these seem to have outliers?

plt.figure(figsize=(10, 6))

plt.hist(df['Size'], bins=20, color='skyblue', edgecolor='black')

plt.title('Histogram for Size')
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# **6. Outlier treatment:**

**Price: From the box plot, it seems like there are some apps with very high price. A price of $200 for an application on the Play Store is very high and suspicious!**

Check out the records with very high price

high_price_apps = df[df['Price'] > 200]

# Display the records
print("Apps with a price above $200:")
print(high_price_apps[['App', 'Price']])

Drop these as most seem to be junk apps

df.dropna(subset=['Price'], inplace=True)
df = df[df['Price'] <= 200]

df.to_csv('GoogleProject.csv', index=False)
print("\nDisplay Data Frame Info:")
print(df.info())

**Reviews:**
- Very few apps have very high number of reviews. These are all star apps that donot help with the analysis and, in fact, will skew it. Drop records having more than 2 million reviews.

high_reviews_apps = df[df['Reviews'] > 1000000]

# Display the apps with high reviews
print("Apps with high reviews:")
print(high_reviews_apps[['App', 'Reviews']])
print("Reviews Description:\n",df['Reviews'].describe())

Dropping these outliers.

df = df[df['Reviews'] <= 1000000]
print("Reviews Description after removing Outliers:\n",df['Reviews'].describe())
df.to_csv('GoogleProject.csv', index=False)
print("\nDisplay Data Frame Info:")
print(df.info())

**Installs:**  
**- There seems to be some outliers in this field too. Apps having very high number of installs should be dropped from the analysis.**

# Find apps with high installs
high_installs_apps = df[df['Installs'] > 100000000]

# Display the apps with high installs
print("Apps with high installs:")
print(high_installs_apps[['App', 'Installs']])
print("Installs Description:\n",df['Installs'].describe())

**- Find out the different percentiles – 10, 25, 50, 70, 90, 95, 99**

percentiles_to_calculate = [10, 25, 50, 70, 90, 95, 99]

install_percentiles = np.percentile(df['Installs'], percentiles_to_calculate)

for p, value in zip(percentiles_to_calculate, install_percentiles):
    print(f"{p}th Percentile: {value}")
print("\nDisplay Data Frame Info:")
print(df.info())

**- Decide a threshold as cutoff for outlier and drop records having values more than that**

df = df[df['Installs'] <= 10000000]
df.to_csv('GoogleProject.csv', index=False)
print("\nDisplay Data Frame Info:")
print(df.info())

print("Installs Description:\n", df['Installs'].describe())

# **7. Bivariate Analysis**
Let’s look at how the available predictors relate to the variable of interest, i.e., our target variable rating. Make scatter plots (for numeric features) and box plots (for character features) to assess the relations between rating and the other features.

**Make scatter plot/joinplot for Rating vs. Price**
- What pattern do you observe? Does rating increase with price?

plt.figure(figsize=(10, 8))

sns.jointplot(x='Price', y='Rating', data=df, kind='scatter', height=8)
plt.suptitle('Joint Plot: Rating vs Price', y=1.02)
plt.show()

# Rating does increase with price.

**Make scatter plot/joinplot for Rating vs. Size**
- Are heavier apps rated better?

sns.jointplot(x='Size', y='Rating', data=df, kind='scatter', height=8)
plt.suptitle('Joint Plot: Rating vs Size', y=1.02)
plt.show()

# Heavy apps have better rating

**Make scatter plot/joinplot for Rating vs. Reviews**
- Does more review mean a better rating always?

sns.jointplot(x='Reviews', y='Rating', data=df, kind='scatter', height=8, ratio=9)
plt.suptitle('Joint Plot: Rating vs Reviews')
plt.show()

# More review means a better rating

**Make boxplot for Rating vs. Content Rating**
- Is there any difference in the ratings? Are some types liked better?

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.boxplot(x='Rating', y='Content Rating', data=df, palette='viridis', showfliers=False)

plt.xlabel('Rating')
plt.ylabel('Content Rating')
plt.title('Boxplot of Rating vs. Content Rating')

plt.show()

# Yes 18+ are liked better.

**Make boxplot for Ratings vs. Category**
- Which genre has the best ratings?

plt.figure(figsize=(18, 9))
sns.boxplot(x='Rating', y='Category', data=df, palette='viridis')

plt.xlabel('Rating')
plt.ylabel('Category')
plt.title('Boxplot of Rating vs. Category')

plt.tight_layout()
plt.show()

# Entertainment has the best rating

# **8. Data preprocessing**
- For the steps below, create a copy of the dataframe to make all the edits. Name it inp1.

df.to_csv('inp1.csv', index=False)
print("\nDisplay Data Frame Info:")
print(df.info())

**- Reviews and Install have some values that are still relatively very high. Before building a linear regression model, you need to reduce the skew. Apply log transformation (np.log1p) to Reviews and Installs.**

df['Reviews'] = np.log1p(df['Reviews'])
df['Installs'] = np.log1p(df['Installs'])

print("Updated DF Info:")
print(df.info())

**- Drop columns App, Last Updated, Current Ver, and Android Ver. These variables are not useful for our task.**

df = df.drop('App', axis=1)
df = df.drop('Last Updated', axis=1)
df = df.drop('Current Ver', axis=1)
df = df.drop('Android Ver', axis=1)

print("Info after dropping columns:")
print(df.info())

**- Get dummy columns for Category, Genres, and Content Rating. This needs to be done as the models do not understand categorical data, and all data should be numeric. Dummy encoding is one way to convert character fields to numeric. Name of dataframe should be inp2.**

# List of columns to create dummy columns for
columns_to_dummy = ['Category', 'Genres', 'Content Rating']

# Create dummy columns
df_dummies = pd.get_dummies(df[columns_to_dummy])

# Concatenate the dummy columns with the original DataFrame
df = pd.concat([df, df_dummies], axis=1)

# Drop the original columns for which dummy columns were created
df.drop(columns=columns_to_dummy, inplace=True)

# Display the updated DataFrame
print(df.head())
print("Info after creating dummy column:")
print(df.info())

print(df.columns)

# Convert 'Installs' column to int
df['Installs'] = df['Installs'].astype(int)
# Round 'Reviews' column to 2 decimals
df['Reviews'] = df['Reviews'].round(4)

# Display the updated DataFrame information
print(df.head())

print(df.info())

df.to_csv('inp2.csv', index=False)

# **9. Train test split  and apply 70-30 split. Name the new dataframes df_train and df_test.**

from sklearn.model_selection import train_test_split

# Target variable (y)
y = df['Rating']

# Train-test split (70-30 ratio)
df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

# **10. Separate the dataframes into X_train, y_train, X_test, and y_test.**

# Features (X) and target variables (y)
X_train = df_train.drop('Rating', axis=1)
y_train = df_train['Rating']

X_test = df_test.drop('Rating', axis=1)
y_test = df_test['Rating']

# Resulting DataFrames
print("Shape of df_train:", df_train.shape)
print("Shape of df_test:", df_test.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# **11 . Model building**

- Use linear regression as the technique
- Report the R2 on the train set

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create a Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# **12. Make predictions on test set and report R2.**

# Predict the target variable on the test set
y_test_pred = model.predict(X_test)

# Calculate R2 score on the test set
r2_test = r2_score(y_test, y_test_pred)

# Print the R2 score on the test set
print("R2 Score on Test Set:", r2_test)

# ***R2 Score on Test Set: 0.1309534673797338***

**Checking Actual vs predicted values**

plt.figure(figsize=(10,10))

plt.scatter(y_test, y_test_pred, c='#F28F2C')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_test_pred), max(y_test))
p2 = min(min(y_test_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'k-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

