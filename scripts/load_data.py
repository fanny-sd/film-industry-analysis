import pandas as pd
import numpy as np
import math
import sys
import kagglehub
from kagglehub import KaggleDatasetAdapter

## Loading functions

def load_dataset(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    return df

def load_dataset_tab_separation(file_path):
    df = pd.read_csv(file_path, low_memory=False, sep='\t')
    return df

def load_dataset_encoding(file_path):
    df = pd.read_csv(file_path, low_memory=False, encoding='latin1')
    return df

## Cleaning functions

def clean_imdb_name_basics(df):
    df = df.dropna(subset=['primaryName']) # Drop rows where primaryName is missing
    df.loc[:,'birthYear'] = pd.to_numeric(df['birthYear'],errors='coerce') # Convert to nullable integer type
    df.loc[:,'birthYear'] = df['birthYear'].astype(pd.Int64Dtype())
    df.loc[:, 'deathYear'] = pd.to_numeric(df['deathYear'],errors='coerce').astype('Int64') # Convert to nullable integer type
    return df

def clean_imdb_title_basics(df):
    df = df.dropna(subset=['primaryTitle']) # Drop rows where primaryTitle is missing
    df.loc[:, 'startYear'] = pd.to_numeric(df['startYear'],errors='coerce').astype('Int64') # Convert to nullable integer type
    df.loc[:, 'endYear'] = pd.to_numeric(df['endYear'],errors='coerce').astype('Int64') # Convert to nullable integer type
    df.loc[:, 'genres'] = df['genres'].str.lower()
    return df
    # note: 'titleType' column shows it contains different types of titles (movies, TV shows, shorts..). I could remove non-movies ..
    
def clean_oscar(df):
    df = df[df['Year'].str.fullmatch(r'\d{4}')] # Keep only rows where Year is a 4-digit number
    df.loc[:, 'Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64') # Convert to nullable integer type
    df = df[df['Year'] >= 1980] # Keep only rows where Year is greater than 1980
    df = df.drop(columns=['Note','Citation']) # Drop columns that are not needed for analysis
    df = df[df['FilmId'].notna()]
    return df

def clean_bafta(df):
    df.loc[:, 'category clean'] = df['category'].str.extract(r'^Film \|\s*(.*?)\s*in \d{4}$')
    df = df[df['year'] >= 1980] # Keep only rows where Year is greater than 1980
    return df

def clean_tmdb(df):
    df = df.dropna(subset=['title']) # Drop rows where title is missing
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce') # Convert to datetime, coercing errors to NaT
    df['release_date'] = df['release_date'].dt.date # Keep only the date part (remove time)
    df = df.drop(columns=['backdrop_path', 'poster_path','overview','homepage','tagline','keywords']) # Drop columns that are not needed for analysis
    df = df.sort_values( # drop duplicate ids
        by='vote_count',      # sort only by vote_count
        ascending=False       # highest vote_count first
    ).drop_duplicates( subset=['id'], keep='first')        # keep the highest-vote_count row
    df = df.sort_values(  # drop duplicate ids
        by='vote_count',      # sort only by vote_count
        ascending=False       # highest vote_count first
    ).drop_duplicates( subset=['title'], keep='first')        # keep the highest-vote_count row
    df['release_year'] = df['release_date'].apply(lambda x: x.year if pd.notna(x) else pd.NA).astype('Int64') # Extract release year from release date and convert to nullable integer type
    df = df[df['release_year'] >= 1980] # Keep only rows where Year is greater than 1980
    #Keep only movies with at least 1000 votes ! Need to reduce file size
    #df = df[df['vote_count'] >= 1000] 
    return df

def clean_box_office_mojo(df):
    column_name1 = ['worldwide_box_office', 'us_box_office', 'foreign_box_office']
    for col in column_name1:
        df[col] = df[col].str.replace('$', '', regex=False)
        df[col] = df[col].str.replace(',', '', regex=False)
        df[col] = df[col].str.replace('-', '', regex=False)
        df[col] = df[col].replace('', np.nan)                # Replace empty strings with NaN so float conversion works
        df[col] = df[col].astype('Int64')
    column_name2 = ['us%', 'foreign%']
    for col in column_name2:
        df[col] = df[col].str.replace('%', '', regex=False)
        df[col] = df[col].str.replace('-', '', regex=False)
        df[col] = df[col].str.replace('<', '', regex=False)
        df[col] = df[col].replace('', np.nan)                 # Replace empty strings with NaN so float conversion works
        df[col] = df[col].astype(float) / 100
        
    # unsure: df['title'] = df['title'].str.lower() 
    # consider: dropping al entries that have low or 0 box office revenue.
    
    df = ( df.sort_values(by='worldwide_box_office', ascending=False) # duplicates
      .drop_duplicates(subset=['title', 'release_year'], keep='first')
      .reset_index(drop=True)) #keep entry with highest worldwide box office for each title + release year combination, and drop the rest 
    
    return df

def clean_the_numbers(df):
    column_name = ['Opening Weekend Revenue','Domestic Box Office','Infl. Adj. Dom. Box Office','International Box Office','Worldwide Box Office','Production Budget','Opening Weekend Theaters','Maximum Theaters','Theatrical Engagements']
    for col in column_name:
        df[col] = df[col].str.replace('$', '', regex=False)
        df[col] = df[col].str.replace(',', '', regex=False)
        df[col] = df[col].astype('Int64')
    # Remove corrupted characters like �, â€”, etc.
    df.loc[:, 'Title'] = (df['Title'].str.replace(r'[^\w\s:.\-]', ' ', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip())
    
    # unsure: df['title'] = df['title'].str.lower() 
    # consider: converting Released and Released Worlwide to dates (I already have release year so might not need to)
    # consider: dropping al entries that have low or 0 box office revenue.
    return df

def clean_netflix_revenue_subs_spend(df):
    cols = ['Netflix Subscribers','Netflix Subscribers (US and Canada)','Netflix Subscribers (EMEA)','Netflix Subscribers (LATAM)','Netflix Subscribers (APAC)','Netflix Content Spending','Netflix Revenue']
    for col in cols:
        df[col] = (df[col]
                .astype(str)
                .str.lower()
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.replace('thousand', '*1e3', regex=False)
                .str.replace('million',  '*1e6', regex=False)
                .str.replace('billion',  '*1e9', regex=False)
                .str.replace(' ', '', regex=False) )
        # convert expressions like "35.89*1e6" into numbers
        df[col] = df[col].replace('', np.nan)                  # handle empty cells
        df[col] = df[col].map(lambda x: eval(x) if (isinstance(x, str) and x != '' and x.lower() != 'nan') else np.nan)
        df[col] = df[col].round().astype('Int64')  # round and convert to integer
    return df

def clean_netflix_engagement(df):
    # consider: changing Title Name to lower case
    # if needed: change 'Runtime' to a duration in minutes (currently in format "1h 30m")
    cols = ['2023 Hours','2023 Views','2024 Hours','2024 Views','2025 Hours','2025 Views','Total Hours Viewed','Total Views']
    for col in cols:
        df[col] = df[col].str.replace('-', '', regex=False)
        df[col] = df[col].str.replace(',', '', regex=False)
        df[col] = df[col].replace('', np.nan)                # Replace empty strings with NaN so float conversion works
        df[col] = df[col].astype('Int64')
    
    # consider: getting rid of low or 0 Total Hours Viewed and Total Views
    return df


if __name__ == "__main__":
    
    ### IMPORTS
    
    # IMDB imports - NOT BEING USED IN THE ANALYSIS, BUT I MIGHT USE IT IN THE FUTURE
    #df_imdb_name_basics = load_dataset('data/imdb_datasets/name.basics.csv')
    #df_imdb_title_basics = load_dataset('data/imdb_datasets/title.basics.csv')
    #df_imdb_title_akas = load_dataset('data/imdb_datasets/title.akas.csv')
    #df_imdb_title_crew = load_dataset('data/imdb_datasets/title.crew.csv')
    #df_imdb_title_episode = load_dataset('data/imdb_datasets/title.episode.csv')
    #df_imdb_title_principals = load_dataset('data/imdb_datasets/title.principals.csv')
    #df_imdb_title_ratings = load_dataset('data/imdb_datasets/title.ratings.csv')
    
    # OSCAR import
    df_oscar = load_dataset_tab_separation('data/oscars_1927-2025/full_data.csv')
    
    # BAFTA import
    df_bafta = load_dataset('data/bafta_1949-2020/bafta_films.csv')
    
    # TMDB import
    df_tmdb = load_dataset('data/tmdb_dataset/TMDB_movie_dataset_v11.csv')
    
    # Box Office Mojo import
    df_box_office_mojo = load_dataset_encoding('data/box_office_mojo/box_office_mojo_2015-2025.csv')
    
    # The Numbers import
    df_the_numbers = load_dataset_encoding('data/the_numbers/the_numbers_box_office_2015-2025.csv')
    
    # Netflix imports
    df_netflix_revenue_subs_spend = load_dataset('data/netflix/netflix_rev_subs_spend.csv')
    df_netflix_engagement = load_dataset('data/netflix/netflix_engagement-report_2023-2025.csv')
    
    
    ### CLEANING
    
    # IMDB cleaning - DECIDED NOT TO USE IMDB DATA IN THE ANALYSIS, BUT I MIGHT USE IT IN THE FUTURE
    #df_imdb_name_basics_cleaned = clean_imdb_name_basics(df_imdb_name_basics)
    #df_imdb_title_basics_cleaned = clean_imdb_title_basics(df_imdb_title_basics)
    # df_imdb_title_akas : no cleaning needed / relevant. There is duplcation in the titleId column, but this is expected as it contains multiple entries for the same title in different regions/languages.
    # df_imdb_title_crew: no cleaning needed. It just links title ids with name ids for directors and writers.
    # df_imdb_title_episode: no cleaning needed. It just links episode titles with their parent series.
    # df_imdb_title_principals: no cleaning needed. It just links title ids with name ids for actors and other crew. I could improve the formatting of the 'characters' column (e.g. ["Sílvia"]), but I don't think it's necessary for my analysis.
    # df_imdb_title_ratings: no cleaning needed.
    
    # OSCAR cleaning
    df_oscar_cleaned = clean_oscar(df_oscar)
    # dropped older years
    
    # BAFTA cleaning
    df_bafta_cleaned = clean_bafta(df_bafta)
    
    # TMDB cleaning
    df_tmdb_cleaned = clean_tmdb(df_tmdb)
    
    # Box Office Mojo cleaning
    df_box_office_mojo_cleaned = clean_box_office_mojo(df_box_office_mojo)
    
    # The Numbers cleaning
    df_the_numbers_cleaned = clean_the_numbers(df_the_numbers)
    
    # Netflix cleaning
    df_netflix_revenue_subs_spend_cleaned = clean_netflix_revenue_subs_spend(df_netflix_revenue_subs_spend)
    df_netflix_engagement_cleaned = clean_netflix_engagement(df_netflix_engagement)

    ### SAVING CLEANED DATA

    #df_imdb_name_basics_cleaned.to_csv('data/clean_data/cleaned_imdb_name_basics.csv', index=False)
    #df_imdb_title_basics_cleaned.to_csv('data/clean_data/cleaned_imdb_title_basics.csv', index=False)
    df_oscar_cleaned.to_csv('data/clean_data/cleaned_oscar.csv', index=False)
    df_bafta_cleaned.to_csv('data/clean_data/cleaned_bafta.csv', index=False)
    df_tmdb_cleaned.to_csv('data/clean_data/cleaned_tmdb.csv', index=False)
    df_box_office_mojo_cleaned.to_csv('data/clean_data/cleaned_box_office_mojo.csv', index=False)
    df_the_numbers_cleaned.to_csv('data/clean_data/cleaned_the_numbers.csv', index=False)
    df_netflix_revenue_subs_spend_cleaned.to_csv('data/clean_data/cleaned_netflix_revenue_subs_spend.csv', index=False)
    df_netflix_engagement_cleaned.to_csv('data/clean_data/cleaned_netflix_engagement.csv', index=False)
    
    
## LIVE CONNECTIONS

#kagglehub.login()
#file_path = "TMDB_movie_dataset_v11.csv"
#df = kagglehub.load_dataset(
#    KaggleDatasetAdapter.PANDAS,
#    "asaniczka/tmdb-movies-dataset-2023-930k-movies",
#    file_path)
#print("First 5 records:", df.head())

