import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import faiss
import torch
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Disable QT GUI
import matplotlib
matplotlib.use('Agg')

def preprocess_hotel_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Data Cleaning
    
    # 1. Handle missing values
    # Replace NULL strings with actual NaN
    df.replace('NULL', pd.NA, inplace=True)
    
    # Fill missing numerical values
    numerical_cols = ['children', 'adults', 'babies', 'days_in_waiting_list']
    for col in numerical_cols:
        df[col] = df[col].fillna(0).astype(int)
    
    # Fill missing categorical values
    categorical_cols = ['country', 'agent', 'company', 'meal']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    # 2. Convert date columns to datetime
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    
    # 3. Convert arrival date month to numerical
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['arrival_date_month'] = df['arrival_date_month'].map(month_map)

    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month'].astype(str) + '-' +
        df['arrival_date_day_of_month'].astype(str),
        errors='coerce'
    )
    
    # 4. Fix data type inconsistencies
    df['hotel'] = df['hotel'].astype('category')
    df['meal'] = df['meal'].astype('category')
    df['country'] = df['country'].astype('category')
    df['market_segment'] = df['market_segment'].astype('category')
    df['distribution_channel'] = df['distribution_channel'].astype('category')
    df['reserved_room_type'] = df['reserved_room_type'].astype('category')
    df['assigned_room_type'] = df['assigned_room_type'].astype('category')
    df['deposit_type'] = df['deposit_type'].astype('category')
    df['customer_type'] = df['customer_type'].astype('category')
    
    # 5. Create total guests column
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    
    # 6. Remove rows with zero guests
    df = df[df['total_guests'] > 0]
    
    # 7. Convert boolean columns
    df['is_canceled'] = df['is_canceled'].astype(bool)
    df['is_repeated_guest'] = df['is_repeated_guest'].astype(bool)

    # 8. Create derived columns
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['revenue'] = df['total_nights'] * df['adr']
    
    # 8. Save cleaned data
    # cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv("processed_data.csv", index=False)  # Consistent filename
    
    return df

def analyze_hotel_data(df):
    """
    Perform analytics on hotel booking data and generate visual reports
    
    Args:
        df (DataFrame): Preprocessed hotel bookings dataframe
    
    Returns:
        dict: Dictionary containing analytical results and figures
    """
    analytics = {}
    
    # 1. Revenue Trends Over Time
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['revenue'] = df['total_nights'] * df['adr']
    
    # Create proper date column
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month'].astype(str) + '-' +
        df['arrival_date_day_of_month'].astype(str)
    )
    
    revenue_trends = df.groupby(pd.Grouper(key='arrival_date', freq='ME'))['revenue'].sum()
    
    plt.figure(figsize=(12, 6))
    revenue_trends.plot(title='Monthly Revenue Trend')
    plt.ylabel('Revenue (€)')
    plt.xlabel('Date')
    analytics['revenue_trends'] = {
        'data': revenue_trends,
        'plot': plt.gcf()
    }
    
    # 2. Cancellation Rate Analysis
    total_bookings = len(df)
    canceled_bookings = df['is_canceled'].sum()
    cancellation_rate = (canceled_bookings / total_bookings) * 100
    
    cancellation_data = pd.DataFrame({
        'Status': ['Canceled', 'Not Canceled'],
        'Count': [canceled_bookings, total_bookings - canceled_bookings]
    })
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Status', y='Count', data=cancellation_data)
    plt.title(f'Cancellation Rate: {cancellation_rate:.2f}%')
    analytics['cancellation_analysis'] = {
        'rate': cancellation_rate,
        'data': cancellation_data,
        'plot': plt.gcf()
    }
    
    # 3. Geographical Distribution
    country_distribution = df['country'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    country_distribution.plot(kind='bar')
    plt.title('Top 10 Countries by Bookings')
    plt.xlabel('Country Code')
    plt.ylabel('Number of Bookings')
    analytics['geographical_distribution'] = {
        'data': country_distribution,
        'plot': plt.gcf()
    }
    
    # 4. Lead Time Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['lead_time'], bins=30, kde=True)
    plt.title('Booking Lead Time Distribution')
    plt.xlabel('Lead Time (Days)')
    plt.ylabel('Count')
    analytics['lead_time_distribution'] = {
        'stats': df['lead_time'].describe(),
        'plot': plt.gcf()
    }
    
    # 5. Additional Analytics: Market Segment Analysis
    segment_distribution = df['market_segment'].value_counts()
    
    plt.figure(figsize=(10, 6))
    segment_distribution.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Market Segment Distribution')
    plt.ylabel('')
    analytics['market_segment_analysis'] = {
        'data': segment_distribution,
        'plot': plt.gcf()
    }
    
    # Close all plots to prevent memory issues
    plt.close('all')
    
    return analytics

def rag_hotel_qa(df, query, use_gpu=False):
    """
    Implement RAG-based Q&A system for hotel bookings data
    
    Args:
        df (DataFrame): Preprocessed hotel bookings dataframe
        query (str): Natural language question
        use_gpu (bool): Whether to use GPU acceleration
    
    Returns:
        dict: Contains answer, context, and confidence
    """
    # Check required columns
    required_cols = ['hotel', 'is_canceled', 'lead_time', 'country', 
                    'adr', 'arrival_date', 'total_nights', 'revenue']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame missing required columns. Run preprocessing first.")
    
    # 1. Prepare data for vector store
    context_data = []
    for _, row in df.iterrows():
        try:
            arrival_date = pd.to_datetime(row['arrival_date']).strftime('%Y-%m-%d')
        except:
            arrival_date = "Unknown date"
            
        context = (
            f"Hotel: {row['hotel']}, "
            f"Status: {'Canceled' if row['is_canceled'] else 'Not Canceled'}, "
            f"Country: {row['country']}, "
            f"Price: {row['adr']} EUR, "
            f"Arrival: {arrival_date}, "
            f"Nights: {row['total_nights']}, "
            f"Revenue: {row['revenue']} EUR"
        )
        context_data.append(context)
    
    # 2. Generate embeddings
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = encoder.encode(context_data, show_progress_bar=True)
    
    # 3. Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(embeddings.astype(np.float32))
    
    # 4. Process query
    query_embedding = encoder.encode([query])
    _, indices = index.search(query_embedding.astype(np.float32), k=5)
    
    # 5. Prepare context for LLM
    context = "\n".join([context_data[i] for i in indices[0]])
    
    # 6. Load LLM
    model_name = "EleutherAI/pythia-12m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto" if use_gpu else None
    )
    
    # 7. Generate response
    prompt = f"""<s>[INST] Answer the question using only this context:
    {context}
    
    Question: {query} [/INST]"""
    
    response = llm(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False
    )
    
    return {
        "question": query,
        "answer": response[0]['generated_text'].strip(),
        "context": context,
        "model": model_name
    }

if __name__ == "__main__":
    # Load and analyze cleaned data
    cleaned_df = pd.read_csv("dataset/hotel_bookings_cleaned.csv")
    results = analyze_hotel_data(cleaned_df)
    
    # To save plots
    # for analysis_name, content in results.items():
    #     if 'plot' in content:
    #         content['plot'].savefig(f"{analysis_name}.png")
    
    questions = [
        "Show me total revenue for July 2017",
        "Which locations had the highest booking cancellations?",
        "What is the average price of a hotel booking?"
    ]
    
    for q in questions:
        result = rag_hotel_qa(cleaned_df, q)
        print(f"\nQuestion: {result['question']}")
        print(f"Answer: {result['answer']}")
        print("="*50)
