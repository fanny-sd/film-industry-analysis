import pandas as pd
import numpy as np
import math
import sys
import re
from load_data import load_dataset, load_dataset_tab_separation, load_dataset_encoding

# PRODUCTION COMPANIES
def prepare_production_companies(df):
    # Split into list
    df = df.copy()
    df['production_companies'] = df['production_companies'].str.lower().str.split(', ')
    # Explode into rows
    df_companies = df.explode('production_companies')
    # Clean spaces after exploding
    df_companies['production_companies'] = (
            df_companies['production_companies'].str.strip())
    # Group similar names
    df_companies['production_companies'] = df_companies['production_companies'].apply(standardise_company)
    return df_companies


def prepare_oscar_production_companies(df):
    df = df.copy()
    # Keep only Oscar winners
    df = df[df['Winner'] == True]
    # Split into list
    df['production_companies'] = df['production_companies'].str.lower().str.split(', ')
    # Explode into rows
    df_companies = df.explode('production_companies')
    # Clean spaces
    df_companies['production_companies'] = (
        df_companies['production_companies'].str.strip())
    # Standardise using your existing function
    df_companies['production_companies'] = df_companies['production_companies'].apply(standardise_company)
    return df_companies


def standardise_company(company):
    if pd.isna(company):
        return company
    if '20th century' in company:
        return '20th century'
    elif 'warner bros' in company:
        return 'warner bros'
    elif 'disney' in company:
        return 'disney'
    elif 'universal' in company:
        return 'universal'
    elif 'paramount' in company:
        return 'paramount'
    elif 'sony' in company:
        return 'sony'
    elif 'columbia pictures' in company:
        return 'columbia pictures'
    elif 'marvel' in company:
        return 'marvel studios'
    elif 'pixar' in company:
        return 'pixar'
    elif 'dreamworks' in company:
        return 'dreamworks'
    elif 'lionsgate' in company: ## next time use 'lions' because there's 3 under 'lions gate films'
        return 'lionsgate'
    elif 'miramax' in company:
        return 'miramax'
    elif 'metro-goldwyn-mayer' in company:
        return 'metro-goldwyn-mayer'
    elif 'legendary pictures' in company:
        return 'legendary pictures'
    elif 'new line cinema' in company:
        return 'new line cinema'
    else:
        return company


# Below not used for Tableau at the moment, but could be useful for future analysis
def production_company_stats(df_companies):
    companies_of_interest = [
        'warner bros','metro-goldwyn-mayer','20th century',
        'marvel studios','paramount','universal','sony','disney',
        'columbia pictures','legendary pictures','pixar',
        'lionsgate','dreamworks','miramax','new line cinema']
    # Filter only companies I care about
    df_filtered = df_companies[
        df_companies['production_companies'].isin(companies_of_interest)]
    # Aggregate
    stats = (
        df_filtered.groupby('production_companies')
        .agg(
            movie_count=('production_companies', 'size'),
            avg_vote_average=('vote_average', 'mean'),
            total_vote_count=('vote_count', 'sum'))
        .reset_index()
        .sort_values('movie_count', ascending=False))
    return stats

# GENRES

def prepare_genres(df):
    df = df.copy()
    df['genres'] = df['genres'].str.lower().str.split(', ')
    df_genres = df.explode('genres')
    df_genres['genres'] = df_genres['genres'].str.strip()
    return df_genres

def prepare_oscar_genres(df):
    df = df.copy()
    # Keep only winners
    df = df[df['Winner'] == True]
    # Split genres
    df['genres'] = df['genres'].str.lower().str.split(', ')
    # Explode into rows
    df_genres = df.explode('genres')
    # Clean spaces
    df_genres['genres'] = df_genres['genres'].str.strip()
    return df_genres

# Below not used for Tableau at the moment, but could be useful for future analysis
def genre_stats(df_genres):
    genres_of_interest = [
        'action','science fiction','comedy','drama','adventure',
        'thriller','fantasy','romance','crime','animation',
        'family','horror','western']
    df_filtered = df_genres[
        df_genres['genres'].isin(genres_of_interest)]
    stats = (
        df_filtered.groupby('genres')
        .agg(
            movie_count=('genres', 'size'),
            avg_vote_average=('vote_average', 'mean'),
            total_vote_count=('vote_count', 'sum')
        ).reset_index().sort_values('movie_count', ascending=False))
    return stats


if __name__ == "__main__":
    # TMDB import
    df_tmdb = load_dataset('data/analysis_data/tmdb_top_movies.csv')
    # Oscar import
    df_oscar = load_dataset('data/analysis_data/merged_oscar_tmdb.csv')
    
    # TMDb
    # take the top 1000 by vote_average
    df_top = df_tmdb.nlargest(1000, 'vote_average')
    df_companies_tmdb = prepare_production_companies(df_top)
    df_genres_tmdb = prepare_genres(df_top)
    
    # Oscars
    df_genres_oscar = prepare_oscar_genres(df_oscar)
    df_companies_oscar = prepare_oscar_production_companies(df_oscar)
    
    # Save the prepared data for Tableau
    df_companies_tmdb.to_csv("data/analysis_data/tmdb_companies_long.csv", index=False)
    df_genres_tmdb.to_csv("data/analysis_data/tmdb_genres_long.csv", index=False)
    df_genres_oscar.to_csv("data/analysis_data/oscar_genres_long.csv", index=False)
    df_companies_oscar.to_csv("data/analysis_data/oscar_companies_long.csv", index=False)

