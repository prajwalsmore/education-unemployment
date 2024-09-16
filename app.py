import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

# Load the preprocessed Excel data
file_path = "preprocessed_education_data.xlsx"
years = ['2016', '2017', '2018', '2019', '2020', '2021']
data = {year: pd.read_excel(file_path, sheet_name=year) for year in years}

# Combine data from all years into a single DataFrame
df_combined = pd.concat(data, keys=years)
df_combined.reset_index(level=0, inplace=True)
df_combined.rename(columns={'level_0': 'Year'}, inplace=True)

# Ensure Year is treated as numeric
df_combined['Year'] = pd.to_numeric(df_combined['Year'], errors='coerce')

# Ensure necessary columns are numeric
df_combined['Estimated Unemployment Rate (%)'] = pd.to_numeric(df_combined['Estimated Unemployment Rate (%)'], errors='coerce')
df_combined['Estimated Employed'] = pd.to_numeric(df_combined['Estimated Employed'], errors='coerce')
df_combined['Estimated Labour Participation Rate (%)'] = pd.to_numeric(df_combined['Estimated Labour Participation Rate (%)'], errors='coerce')
df_combined['UnderGraduate'] = pd.to_numeric(df_combined['UnderGraduate'], errors='coerce')
df_combined['PostGraduate'] = pd.to_numeric(df_combined['PostGraduate'], errors='coerce')

# Streamlit app starts here
st.title('Indian Education and Employment Analysis (2016-2021)')

# Sidebar for selecting the state and year
st.sidebar.header("Filters")

selected_state = st.sidebar.selectbox('Select State', df_combined['State'].unique())
selected_year = st.sidebar.selectbox('Select Year', df_combined['Year'].unique())

# Filter the data based on selection
filtered_data = df_combined[(df_combined['State'] == selected_state) & (df_combined['Year'] == selected_year)]
st.write(f"### Displaying data for **{selected_state}** in **{selected_year}**")
st.dataframe(filtered_data)

# Remove unnecessary inputs from sidebar
# (Assuming these were previously present; based on your request, they are already removed)

# Add a percentage threshold for employment
st.sidebar.header("Employment Filter")
employment_threshold = st.sidebar.slider('Select Employment Threshold (%)', 0, 100, 30)

# Filter states with employment below or equal to the threshold
employment_filtered = df_combined[df_combined['Estimated Labour Participation Rate (%)'] <= employment_threshold]
st.subheader(f'States with Labour Participation Rate ≤ {employment_threshold}%')
st.dataframe(employment_filtered[['State', 'Estimated Labour Participation Rate (%)']])

# Trend of Unemployment Rate over the years for the selected state
st.subheader('Trend of Unemployment Rate (2016-2021)')
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df_combined[df_combined['State'] == selected_state], x='Year', y='Estimated Unemployment Rate (%)', ax=ax)
ax.set_title(f'Unemployment Rate in {selected_state} (2016-2021)')
ax.set_xlabel('Year')
ax.set_ylabel('Unemployment Rate (%)')
st.pyplot(fig)

# User selects which education level to analyze
education_level = st.selectbox('Select Education Level:', ['Below Graduation', 'UnderGraduate', 'PostGraduate', 'Ph.D.'])

# Education trend for the selected state
st.subheader(f'Trend of {education_level} in {selected_state} (2016-2021)')
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df_combined[df_combined['State'] == selected_state], x='Year', y=education_level, ax=ax)
ax.set_title(f'{education_level} Trends in {selected_state} (2016-2021)')
ax.set_xlabel('Year')
ax.set_ylabel(education_level)
st.pyplot(fig)

# Correlation between unemployment and education levels
st.subheader('Correlation between Education Levels and Unemployment')
education_columns = ['Below Graduation', 'UnderGraduate', 'PostGraduate', 'Ph.D.']
education_unemployment_corr = df_combined.corr(numeric_only=True)['Estimated Unemployment Rate (%)'][education_columns]
st.write(education_unemployment_corr)

# Select data to compare Unemployment Category
unemployment_threshold = df_combined['Estimated Unemployment Rate (%)'].mean()
df_combined['Unemployment Category'] = df_combined['Estimated Unemployment Rate (%)'].apply(
    lambda x: 'High' if x > unemployment_threshold else 'Low')

st.subheader('Comparison of Employment Numbers in High vs. Low Unemployment States')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_combined, x='Unemployment Category', y='Estimated Employed', ax=ax)
ax.set_title('Employment Numbers in High vs. Low Unemployment States')
ax.set_xlabel('Unemployment Category')
ax.set_ylabel('Estimated Employed')
st.pyplot(fig)

# Add a slider for year range selection
st.sidebar.header("Year Range")
year_range = st.sidebar.slider('Select Year Range', 2016, 2021, (2016, 2021))

# Filter data for the selected year range
filtered_df = df_combined[(df_combined['Year'] >= year_range[0]) & (df_combined['Year'] <= year_range[1])]

# Create a choropleth map for labor participation rates
st.subheader('Labour Participation Rate Across States')
fig = px.choropleth(
    filtered_df,
    locations='State',
    locationmode='country names',
    color='Estimated Labour Participation Rate (%)',
    hover_name='State',
    color_continuous_scale='Blues',
    title='Labour Participation Rate Across States'
)
st.plotly_chart(fig)

# Machine learning model section
st.sidebar.header("Machine Learning Model")

# Machine learning model selection
model_option = st.sidebar.selectbox('Select a Model', ['Linear Regression'])

# Data preparation for ML models
X = df_combined[['Estimated Employed', 'Estimated Labour Participation Rate (%)', 'UnderGraduate', 'PostGraduate']]
y = df_combined['Estimated Unemployment Rate (%)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model based on selection
if model_option == 'Linear Regression':
    model = LinearRegression()
elif model_option == 'Decision Tree Regressor':
    model = DecisionTreeRegressor(random_state=42)
elif model_option == 'Random Forest Regressor':
    model = RandomForestRegressor(random_state=42)

# Check if a trained model exists
model_path = f'{model_option.replace(" ", "_").lower()}_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.sidebar.success(f'Loaded existing {model_option} model.')
else:
    # Train the selected model
    model.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(model, model_path)
    st.sidebar.success(f'Trained and saved new {model_option} model.')

# Display model performance
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader(f'{model_option} Performance')
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R-squared (R²):** {r2:.2f}")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define threshold to classify unemployment rates
unemployment_threshold = df_combined['Estimated Unemployment Rate (%)'].mean()
df_combined['Unemployment Category'] = df_combined['Estimated Unemployment Rate (%)'].apply(
    lambda x: 1 if x > unemployment_threshold else 0  # 1 for High, 0 for Low
)

# Sidebar for selecting the machine learning model
st.sidebar.header("Machine Learning Model")

# Model selection (classification)
model_option = st.sidebar.selectbox('Select a Model', ['Decision Tree Classifier', 'Random Forest Classifier'])

# Prepare the data for classification
X = df_combined[['Estimated Employed', 'Estimated Labour Participation Rate (%)', 'UnderGraduate', 'PostGraduate']]
y = df_combined['Unemployment Category']  # Binary classification: 1 for High, 0 for Low

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the selected classification model
if model_option == 'Decision Tree Classifier':
    model = DecisionTreeClassifier(random_state=42)
elif model_option == 'Random Forest Classifier':
    model = RandomForestClassifier(random_state=42)

# Check if a trained model exists
model_path = f'{model_option.replace(" ", "_").lower()}_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.sidebar.success(f'Loaded existing {model_option} model.')
else:
    # Train the selected model
    model.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(model, model_path)
    st.sidebar.success(f'Trained and saved new {model_option} model.')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display classification performance metrics
st.subheader(f'{model_option} Performance')
st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1 Score:** {f1:.2f}")

# Correlation Matrix
st.subheader('Correlation Matrix')
numerical_columns = df_combined.select_dtypes(include=['float64', 'int64']).columns
selected_columns = st.multiselect('Select columns for correlation', numerical_columns, default=numerical_columns)
if selected_columns:
    corr_matrix = df_combined[selected_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Allow users to download the filtered data
st.subheader('Download Filtered Data')
csv = filtered_data.to_csv(index=False).encode('utf-8')
st.download_button(label='Download Filtered Data', data=csv, file_name='filtered_data.csv', mime='text/csv')

# Feedback section
st.subheader('Feedback')
feedback = st.text_area("Leave your feedback or insights here!")
if st.button("Submit Feedback"):
    # Here, you might want to save the feedback to a file or database
    st.write("Thank you for your feedback!")

# Display key insights
st.subheader('Key Insights')
highest_unemployment = df_combined.groupby('State')['Estimated Unemployment Rate (%)'].mean().idxmax()
lowest_unemployment = df_combined.groupby('State')['Estimated Unemployment Rate (%)'].mean().idxmin()
st.write(f"**Highest unemployment rate:** {highest_unemployment}")
st.write(f"**Lowest unemployment rate:** {lowest_unemployment}")
