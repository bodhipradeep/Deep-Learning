from tensorflow.keras.models import load_model
import re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import matplotlib.pyplot as plt
from text_cleaning import preprocess_text

st.header('IT Ticket Analysis and Classification')
st.write(':blue[Model accuracy ranges from 85% to 95% depending on data quality.]')

# Load the pre-trained model
try:
    model = load_model('model.h5')
    st.success('Model is successfully loaded and ready for data analysis!')
except Exception as e:
    st.error(f"Failed to load model: {e}")

# File uploader for user to upload a CSV or Excel file
file = st.file_uploader('Upload file as CSV or Excel format', type=['csv', 'xlsx'])

if file is not None:
    try:
        # Read the uploaded file based on its extension
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, header=0)
            st.success('CSV file successfully loaded!')
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
            st.success('Excel file successfully loaded!')
        
        # Display first few rows of the dataframe
        st.write('Here is a preview of your data:')
        st.dataframe(df.head()) 
        st.write(f"Data Shape: {df.shape}")

        # Let user select the column for prediction
        column = st.selectbox('Select the Issue/Symptom Column:', ['Choose column'] + list(df.columns))

        if column != 'Choose column':
            st.write(f'You selected the column: **{column}**')
            
            
            # Apply preprocessing
            df[column] = df[column].astype('str').apply(preprocess_text)

            # Tokenization and padding
            max_features = 5000
            max_len = 150
            tokenizer = Tokenizer(num_words=max_features, split=' ')
            tokenizer.fit_on_texts(df[column].values)
            X = tokenizer.texts_to_sequences(df[column].values)
            X = pad_sequences(X, maxlen=max_len)

            # Show spinner while processing the predictions
            with st.spinner('Analyzing data and will predict soon...'):
                # Perform prediction
                pred = model.predict(X)
        
            # Load category mapping
            cat = pd.read_csv('Cat.csv')
            categories = list(cat['Main Category'])  # Ensure 'Main Category' exists
            
            # Reverse one-hot encoding
            Y_reversed = [categories[np.argmax(row)] for row in pred]

            # Convert predictions to DataFrame
            df_reversed = pd.DataFrame(Y_reversed, columns=['Main Category'])
            
            # Count and percentage for each category
            predicted_counts = df_reversed['Main Category'].value_counts()
            predicted_percentages = (predicted_counts / len(df_reversed)) * 100
            results = pd.DataFrame({
                'Category': predicted_counts.index,
                'Count': predicted_counts.values,
                'Percentage': np.round(predicted_percentages.values, 2)
            })
            
            st.write('Predicted Category Distribution:')
            st.dataframe(results)
            
            # Plotting a column chart
            fig, ax = plt.subplots(figsize=[10, 10])
            bars = ax.barh(results['Category'].astype(str), results['Count'], color='skyblue')
            ax.set_title('Predicted Category Distribution', fontsize=15)
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            
            # Annotate bars with percentage
            for bar, percentage in zip(bars, results['Percentage']):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                        f'{percentage:.1f}%', ha='left', va='center', fontsize=10)
            
            # Adjust the font size for the tick labels
            ax.tick_params(axis='x', labelsize=10)  # X-axis tick labels
            ax.tick_params(axis='y', labelsize=10)  # Y-axis tick labels
            
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info('Please upload a file to proceed.')
