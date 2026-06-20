import pandas as pd
import numpy as np
import math
import sys
import re
from load_data import load_dataset, load_dataset_tab_separation, load_dataset_encoding

def clean_title(title):
    # Remove all non-word characters (punctuation and whitespace)
    return re.sub(r'[^\w]', '', str(title)).lower().strip()

def merge_left_join(df1, df2, column_df1, column_df2):
    # Create temporary join columns
    #left_key = df1[column_df1].astype(str).str.strip().str.lower()
    #right_key = df2[column_df2].astype(str).str.strip().str.lower()
    return pd.merge(df1, df2, left_on=column_df1, right_on=column_df2, how='left', indicator=True)

def merge_outer_join(df1, df2, column_df1, column_df2, indicator):
    # Assuming columns are already cleaned externally
    return pd.merge(df1, df2, left_on=column_df1, right_on=column_df2, how='outer', indicator=indicator)


if __name__ == "__main__":
    
    ### Load cleaned datasets using functions from load_data.py
    
    # IMDB imports - DECIDED NOT TO USE IMDB DATA IN THE ANALYSIS, BUT I MIGHT USE IT IN THE FUTURE
    #df_imdb_name_basics_c = load_dataset('data/clean_data/cleaned_imdb_name_basics.csv')
    #df_imdb_title_basics_c = load_dataset('data/clean_data/cleaned_imdb_title_basics.csv') 
    #df_imdb_title_ratings_c = load_dataset('data/imdb_datasets/title.ratings.csv') # no clean file
    # OSCAR import
    df_oscar_c = load_dataset('data/clean_data/cleaned_oscar.csv')
    # BAFTA import
    df_bafta_c = load_dataset('data/clean_data/cleaned_bafta.csv')
    # TMDB import
    df_tmdb_c = load_dataset('data/clean_data/cleaned_tmdb.csv')
    # Box Office Mojo import
    df_box_office_mojo_c = load_dataset_encoding('data/clean_data/cleaned_box_office_mojo.csv')
    # The Numbers import
    df_the_numbers_c = load_dataset_encoding('data/clean_data/cleaned_the_numbers.csv')
    # Netflix imports
    df_netflix_revenue_subs_spend_c = load_dataset('data/clean_data/cleaned_netflix_revenue_subs_spend.csv')
    df_netflix_engagement_c = load_dataset('data/clean_data/cleaned_netflix_engagement.csv')
    
    
    ### MERGING
    
    # TMDB and IMDB ratings - decided to drop this merge - match percentage is not good enough and TMDB has genres and other useful information
    #df_tmdb__imdb_ratings_m = merge_left_join(df_tmdb_c, df_imdb_title_ratings_c, 'imdb_id', 'tconst')
    #df_tmdb__imdb_ratings_m = df_tmdb__imdb_ratings_m.drop(columns=['original_title','tconst','_merge','key_0']) 
    
    
    ### Box Office Mojo & The Numbers & TMDB
    
    ## BOX OFFICE MOJO - Match percentage for df_1: 95.27%
    
    # Mojo - add prefix to df columns (except the join column)
    merge_column_df1 = 'title'
    df_box_office_mojo_c_prefixed = df_box_office_mojo_c.rename(columns=lambda c: f"mojo_{c}" if c != merge_column_df1 else c)
    
    # Save originals before cleaning
    original_mojo_titles = df_box_office_mojo_c_prefixed['title'].copy()
    original_tmdb_titles = df_tmdb_c['title'].copy()
    
    # Clean titles for merging
    df_box_office_mojo_c_prefixed['title'] = df_box_office_mojo_c_prefixed['title'].apply(clean_title)
    df_tmdb_c['title'] = df_tmdb_c['title'].apply(clean_title)
    
    # OUTER merge:
    df_tmdb__box_office_mojo_m = merge_outer_join(df_box_office_mojo_c_prefixed ,df_tmdb_c, 'title', 'title','merge_12')
    df_tmdb__box_office_mojo_m = df_tmdb__box_office_mojo_m.drop(columns=['original_title','imdb_id']) 
    
    # Restore original titles
    # NOTE: CHECK that this doesn't cause an issue with Outer merges ! It did for Left join, so copy the code for Oscars if needed
    df_box_office_mojo_c_prefixed['title'] = original_mojo_titles
    df_tmdb_c['title'] = original_tmdb_titles
    
    ## THE NUMBERS - Match percentage for df_1: 89.18%
    # The Numbers - Add prefix to df columns (except the join column)
    merge_column_df1 = 'Title'
    df_the_numbers_c_prefixed = df_the_numbers_c.rename(columns=lambda c: f"num_{c}" if c != merge_column_df1 else c)
    # Save originals before cleaning
    original_numbers_titles = df_the_numbers_c_prefixed['Title'].copy()
    original_merged_titles = df_tmdb__box_office_mojo_m['title'].copy()
    # Clean titles for merging
    df_the_numbers_c_prefixed['Title'] = df_the_numbers_c_prefixed['Title'].apply(clean_title)
    df_tmdb__box_office_mojo_m['title'] = df_tmdb__box_office_mojo_m['title'].apply(clean_title)
    # OUTER merge:
    df_tmdb__box_office_mojo_the_numbers_m = merge_outer_join(df_the_numbers_c_prefixed, df_tmdb__box_office_mojo_m, 'Title', 'title','merge_123')
    # Restore original titles
    df_the_numbers_c_prefixed['Title'] = original_numbers_titles
    df_tmdb__box_office_mojo_m['title'] = original_merged_titles
    #Last step: Drop rows with no real match
    df_tmdb__box_office_mojo_the_numbers_m = df_tmdb__box_office_mojo_the_numbers_m[
        (df_tmdb__box_office_mojo_the_numbers_m['merge_12'] == 'both') |
        (df_tmdb__box_office_mojo_the_numbers_m['merge_123'] == 'both') ]
    # drop useless columns:
    #df_tmdb__box_office_mojo_the_numbers_m  = df_tmdb__box_office_mojo_the_numbers_m .drop(columns=['key_0']) 
    
    # ISSUE 
    # FUTURE TASK: MERGE NUMBERS & MOJO and compare the figures for box office !
    # Approach? If the difference is not too big, take the biggest one ?
    # https://www.reddit.com/r/boxoffice/comments/15da6rs/box_office_mojo_vs_thenumbers/?rdt=42851
    
    
    # Bafta & TMDB
    # Clean titles for merging
    df_bafta_c['nominee_clean'] = df_bafta_c['nominee'].apply(clean_title)
    df_tmdb_c['title_clean'] = df_tmdb_c['title'].apply(clean_title)
    # Use clean title columns for merging
    df_bafta__tmdb_m = merge_left_join(df_bafta_c, df_tmdb_c, 'nominee_clean', 'title_clean')
    
    #Notes: There is duplication of the nominees in the Bafta dataset, probably because there are multiple nominations for the same film.
    # it could also be because of TMDb - there is a one to many relationship between films (happened with the Oscars too)
    #Note: Some of the unmatched rows in the Bafta datasets are due to nominees being people, not just films.
    
    # OSCAR & TMDB
    # Clean titles for merging
    df_oscar_c['Film_clean'] = df_oscar_c['Film'].apply(clean_title)
    df_tmdb_c['title_clean'] = df_tmdb_c['title'].apply(clean_title)
    # Use clean title columns for merging
    df_oscar__tmdb_m = merge_left_join(df_oscar_c, df_tmdb_c, 'Film_clean', 'title_clean') 
    # Deduplicate: keep only the TMDb entry with the highest vote_count for each unique Oscar nomination
    dedup_cols = ['Ceremony', 'CanonicalCategory', 'Category', 'Name', 'Film']
    if 'vote_count' in df_oscar__tmdb_m.columns:
        df_oscar__tmdb_m = df_oscar__tmdb_m.sort_values('vote_count', ascending=False)
        # Only use columns that exist in the DataFrame
        dedup_cols_actual = [col for col in dedup_cols if col in df_oscar__tmdb_m.columns]
        df_oscar__tmdb_m = df_oscar__tmdb_m.drop_duplicates(subset=dedup_cols_actual, keep='first')
    # drop useless columns:
    df_oscar__tmdb_m  = df_oscar__tmdb_m .drop(columns=['title_clean','imdb_id','status','id','Film_clean','NomineeIds','FilmId','NomId'])
    # Match percentage for df_1: 110.95% ... 
    # Unmatched rows for oscars: 249 / 5059 !
    # Note - to improve match even more, could try using 'contains' (oscar name contains tmdb name, not the other way around)
    
    # TMDB - REDUCING FILE SIZE FOR ANALYSIS
    df_tmdb_top_movies = df_tmdb_c[df_tmdb_c['vote_count'] >= 1000]  # Keep only movies with at least 1000 votes
    
    ### CHECK Merge Results
    
    df = df_oscar__tmdb_m # Select the merged dataframe you want to check
    df1 = df_oscar_c
    df2 = df_tmdb_c

    #matched_rows = df[df['merge_123'] == 'both'].shape[0]
    matched_rows = df[df['_merge'] == 'both'].shape[0]
    # Total number of rows in both dataframes
    total_df1_rows = df1.shape[0]
    total_df2_rows = df2.shape[0]

    # Calculate match percentage for each dataframe
    match_percentage_df1 = (matched_rows / total_df1_rows) * 100
    match_percentage_df2 = (matched_rows / total_df2_rows) * 100
    
    #matched_rows_sample = matched_rows[['title','vote_count','numVotes']].head(20).to_string()
    #print("\nSample of matched rows:")
    #print(matched_rows_sample) 
    #print("\nMatched rows:")
    #print(matched_rows.head(5).to_string())

    # Display results
    print(f"Total rows in df_1: {total_df1_rows}")
    print(f"Total rows in df_2: {total_df2_rows}")
    print(f"Matched rows: {matched_rows}")
    print(f"Match percentage for df_1: {match_percentage_df1:.2f}%")
    print(f"Match percentage for df_2: {match_percentage_df2:.2f}%")

    #print(df['key_0'].head(20).to_string(), flush=True)
    print(df.dtypes.to_string())

    
    ### SAVING MERGED DATA
    
    #df_tmdb__imdb_ratings_m.to_csv('data/analysis_data/merged_tmdb_imdb_ratings.csv', index=False)
    #df_tmdb_imdb__box_office_mojo_m.to_csv('data/analysis_data/merged_tmdb_imdb_box_office_mojo.csv', index=False)
    df_bafta__tmdb_m.to_csv('data/analysis_data/merged_bafta_tmdb.csv', index=False)
    df_oscar__tmdb_m.to_csv('data/analysis_data/merged_oscar_tmdb.csv', index=False)
    df_tmdb__box_office_mojo_the_numbers_m.to_csv('data/analysis_data/merged_tmdb_box_office_mojo_the_numbers.csv', index=False)
    
    ### SAVING OTHER DATA TO ANALYSIS DATA FOLDER
    
    df_tmdb_top_movies.to_csv('data/analysis_data/tmdb_top_movies.csv', index=False)
    df_the_numbers_c.to_csv('data/analysis_data/cleaned_the_numbers.csv', index=False)
    df_netflix_engagement_c.to_csv('data/analysis_data/cleaned_netflix_engagement.csv', index=False)
    df_netflix_revenue_subs_spend_c.to_csv('data/analysis_data/cleaned_netflix_revenue_subs_spend.csv', index=False)
