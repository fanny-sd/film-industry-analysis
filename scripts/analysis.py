import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import sys
import os
import dataframe_image as dfi
# Add parent directory to path to allow imports from sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_dataset, load_dataset_tab_separation, load_dataset_encoding


### TMDB

def tmdb_top_1000(df):
    df = df.drop(columns=['id','imdb_id','popularity']) # Drop columns that are not needed for analysis
    #Keep only movies with at least 1000 votes
    # MOVING THIS TO CLEANING STEP: df_filtered = df[df['vote_count'] >= 1000]
    # take the top 1000 by vote_average
    df_top = df.nlargest(1000, 'vote_average')
    # Genres
    genres_of_interest = ['action', 'science fiction', 'comedy', 'drama', 'adventure', 'thriller', 'fantasy', 'romance', 'crime', 'animation', 'family', 'horror', 'western']
    # Create dummy columns for Genres (1 if genre in row, 0 otherwise; case-insensitive)
    for genre in genres_of_interest:
        df_top[genre] = df_top['genres'].str.lower().str.contains(genre.lower(), na=False).astype(int)
    # Production Companies
    production_companies = ['warner bros','metro-goldwyn-mayer', '20th century', 'marvel studios', 'paramount', 'universal','sony','disney','columbia pictures','legendary pictures','pixar','lionsgate','dreamworks','miramax','new line cinema']
    # Create dummy columns per company (1 if company in row, 0 otherwise; case-insensitive)
    for company in production_companies:
        df_top[company] = df_top['production_companies'].str.lower().str.contains(company.lower(), na=False).astype(int)
    return df_top

def popular_genres_top_1000(df, save_fig=False):
    genres_of_interest = ['action', 'science fiction', 'comedy', 'drama', 'adventure', 'thriller', 'fantasy', 'romance', 'crime', 'animation', 'family', 'horror', 'western']

    # Melt Genre data to long format for aggregation (one row per movie-genre pair)
    df_melted = df.melt(id_vars=['vote_count', 'vote_average'],
                        value_vars=genres_of_interest,
                        var_name='genre', value_name='present')
    df_melted = df_melted[df_melted['present'] == 1].drop(columns='present')

    # Aggregate per genre:
    genre_stats = df_melted.groupby('genre').agg(
        average_vote=('vote_average', 'mean'),
        movie_count=('genre', 'size')
    ).reset_index()
    genre_stats['average_vote'] = genre_stats['average_vote'].round(2) # round to 2 decimal places
    # sort table by average_vote in descending order
    genre_stats = genre_stats.sort_values('average_vote', ascending=False).reset_index(drop=True)

    # Bar chart
    num_top_movies = len(df)
    genre_stats_sorted = genre_stats.sort_values('movie_count', ascending=False).reset_index(drop=True)
    metric = 'movie_count'
    y_label = 'Number of Movies'
    title = 'Most Popular Genres by share of Top 1000 Movies'

    fig, ax = plt.subplots(figsize=(12, 8))
    positions = range(len(genre_stats_sorted))

    bars = ax.bar(
        positions,
        genre_stats_sorted[metric],
        color=sns.color_palette('viridis', len(genre_stats_sorted)))

    ax.set_xticks(positions)
    ax.set_xticklabels(genre_stats_sorted['genre'], rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel('Genre')
    ax.set_ylabel(y_label)
    ax.set_ylim(0, genre_stats_sorted[metric].max() * 1.1)

    for bar, count in zip(bars, genre_stats_sorted['movie_count']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{count:.0f}',
            ha='center',
            va='bottom',
            fontsize=9)

    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()

    if save_fig:
        fig.savefig('../outputs/tmdb_popularity_analysis/popular_genres_top_1000.png',
                    dpi=300, bbox_inches='tight')
    # Table
    fig_table, ax_table = plt.subplots(figsize=(10, 4))
    ax_table.axis('off')

    table = ax_table.table(
        cellText=genre_stats.values,
        colLabels=genre_stats.columns,
        loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    fig_table.savefig('../outputs/tmdb_popularity_analysis/popular_genres_top_1000_table.png',
                      dpi=300, bbox_inches='tight')
    plt.show()
    return print(genre_stats.to_string(index=False))


def main_production_companies_top_1000(df, save_fig=False):
    # Production companies of interest
    production_companies = ['warner bros','metro-goldwyn-mayer', '20th century', 'marvel studios', 'paramount', 'universal','sony','disney','columbia pictures','legendary pictures','pixar','lionsgate','dreamworks','miramax','new line cinema']
    # Melt the data to long format for aggregation (one row per movie-production_company pair)
    df_melted = df.melt(id_vars=['vote_count', 'vote_average'],
                        value_vars=production_companies,
                        var_name='production_company',
                        value_name='present')
    df_melted = df_melted[df_melted['present'] == 1].drop(columns='present')
    # Aggregate per production company
    company_stats = df_melted.groupby('production_company').agg(
        average_vote=('vote_average', 'mean'),
        movie_count=('production_company', 'size')
    ).reset_index()
    
    company_stats['average_vote'] = company_stats['average_vote'].round(2) # round to 2 decimal places
    # Sort table by average_vote in descending order
    company_stats = company_stats.sort_values('average_vote', ascending=False).reset_index(drop=True)

    # Bar chart
    company_stats_sorted = company_stats.sort_values('movie_count', ascending=False).reset_index(drop=True)
    metric = 'movie_count'
    y_label = 'Number of Movies'
    title = 'Most Popular Production Companies by Number of Movies in Top 1000'
    fig, ax = plt.subplots(figsize=(12, 8))
    positions = range(len(company_stats_sorted))

    bars = ax.bar(
        positions,
        company_stats_sorted[metric],
        color=sns.color_palette('plasma', len(company_stats_sorted)))

    ax.set_xticks(positions)
    ax.set_xticklabels(company_stats_sorted['production_company'], rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel('Production Company')
    ax.set_ylabel(y_label)
    ax.set_ylim(0, company_stats_sorted[metric].max() * 1.1)

    for bar, count in zip(bars, company_stats_sorted['movie_count']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{count:.0f}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()

    # Save figure to output folder
    if save_fig:
        fig.savefig('../outputs/tmdb_popularity_analysis/main_production_companies_top_1000.png',
                    dpi=300, bbox_inches='tight')
    # Save table to output folder
    fig_table, ax_table = plt.subplots(figsize=(10, 4))
    ax_table.axis('off')

    table = ax_table.table(
        cellText=company_stats.values,
        colLabels=company_stats.columns,
        loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    fig_table.savefig('../outputs/tmdb_popularity_analysis/main_production_companies_top_1000_table.png',
                      dpi=300, bbox_inches='tight')
    plt.show()
    return print(company_stats.to_string(index=False))


def top20_movies_by_average_vote(df,start,end, save_fig=False):
    df_tmdb_top_1000 = df[(df['release_year'] >= start) & (df['release_year'] <= end)]
    top_titles = df_tmdb_top_1000.sort_values('vote_average', ascending=False).head(20).copy()
    # Assigning a primary genre for color-coding if a title has any genre flag.
    # Note: f multiple genre flags are present, the first match in genres_of_interest is used - so the order of genres_of_interest determines the primary genre.
    genres_of_interest = ['action', 'science fiction','adventure','drama','romance','fantasy','animation', 'comedy', 'horror', 'thriller','western','family','crime']
    def primary_genre(row):
        for genre in genres_of_interest:
            if row.get(genre, 0) == 1:
                return genre
        return 'other'
    top_titles['primary_genre'] = top_titles.apply(primary_genre, axis=1)
    # Build a palette for the primary genres
    palette_keys = genres_of_interest + ['other']
    genre_palette = dict(zip(palette_keys, sns.color_palette('tab20', len(palette_keys))))
    colors = [genre_palette[g] for g in top_titles['primary_genre']]
    colors = colors[::-1]  # Reverse colors to match reversed top_titles
    # Horizontal bar chart: title on y-axis, average vote on x-axis
    fig, ax = plt.subplots(figsize=(8, 8))
    top_titles = top_titles[::-1]  # reverse for correct ordering in horizontal bar chart
    bars = ax.barh(top_titles['title'], top_titles['vote_average'], color=colors)
    ax.set_xlabel('Average Vote Average')
    ax.set_ylabel('Movie Title')
    ax.set_title('Top 20 Movies by Average Vote (/10)')
    ax.set_xlim(6, top_titles['vote_average'].max() * 1.05)
    for bar, avg in zip(bars, top_titles['vote_average']):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f'{avg:.2f}', va='center', fontsize=9)
    # Legend for the primary genres actually used in this plot
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=genre_palette[g], label=g) for g in palette_keys if g in top_titles['primary_genre'].values]
    ax.legend(handles=legend_handles, title='Primary Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    fig.tight_layout()
    # Save figure to output folder
    if save_fig:
        fig.savefig(f'../outputs/tmdb_popularity_analysis/top20_movies_by_average_vote_{start}-{end}.png', dpi=300, bbox_inches='tight')
    plt.show()



### BOX OFFICE

def worldwide_box_office_trend_by_distributor(df, show=True, save_fig=False):
    # group by distributor and worldwide to get top distributors
    distributors_worldwide_totals = df.groupby('Theatrical Distributor')['Worldwide Box Office'].sum().sort_values(ascending=False)
    top_distributors = distributors_worldwide_totals.head(5).index
    # Filter df for those distributors
    df_top_distributors = df[df['Theatrical Distributor'].isin(top_distributors)]
    # Group by release year and distributor, sum worldwide box office
    df_boxoffice_by_distributor_year = df_top_distributors.groupby(['Release Year', 'Theatrical Distributor'])['Worldwide Box Office'].sum().reset_index()
    # Plot line chart
    fig, ax = plt.subplots(figsize=(12, 8))
    years = sorted(df_boxoffice_by_distributor_year['Release Year'].unique()) # Get sorted unique years for x-axis
    sns.lineplot(data=df_boxoffice_by_distributor_year, x='Release Year', y='Worldwide Box Office', hue='Theatrical Distributor', marker='o', ax=ax)
    ax.set_title('Worldwide Box Office Trend by Top 5 Theatrical Distributors', fontsize=16)
    ax.set_xticks(years) # Set x-ticks to be the unique years
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5) # Add some padding to x-axis limits
    ax.set_xlabel('Release Year', fontsize=12)
    ax.set_ylabel('Worldwide Box Office (Billions USD)', fontsize=12)
    ax.legend(title='Theatrical Distributor')
    ax.grid(axis='x', color='lightgrey', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', color='lightgrey', linestyle='--', linewidth=0.5)
    # Format y-axis to show in billions
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x/1e9:.1f}B'))
    # Save figure to output folder
    if save_fig:
        fig.savefig('../outputs/box_office_analysis/worldwide_box_office_trend_by_distributor.png', dpi=300, bbox_inches='tight')
    plt.show()
    

def movie_count_per_distributor_overtime_top8(df, show=True, save_fig=False):
    overall_top_theatrical_distributors = df['Theatrical Distributor'].value_counts().head(8).index
    #Filter for these distributors
    df_top_distributors=df[df['Theatrical Distributor'].isin(overall_top_theatrical_distributors)]
    #Group by year and distributors, then count the number of movies
    trend_count = df_top_distributors.groupby(['Release Year','Theatrical Distributor']).size().reset_index(name='Movie Count'  )
    #Pivot the data for plotting - years as index, distributors as columns, and movie count as values
    trend_count_pivot = trend_count.pivot(index='Release Year', columns='Theatrical Distributor', values='Movie Count').fillna(0)

    # Prepare stable color mapping based on overall totals (so colors are consistent across years)
    distributor_totals = trend_count_pivot.sum().sort_values(ascending=False)
    distributors_ordered_by_total = distributor_totals.index.tolist()
    distributors = trend_count_pivot.columns.tolist()
    palette = sns.color_palette('tab10', n_colors=len(distributors))
    color_map = {dist: palette[distributors_ordered_by_total.index(dist) % len(palette)] for dist in distributors}

    # Plot stacked bars where each year's stack order is sorted by that year's values (descending)
    fig, ax = plt.subplots(figsize=(14, 10))
    years = list(trend_count_pivot.index)
    x = np.arange(len(years))
    width = 0.8

    for i, year in enumerate(years):
        row = trend_count_pivot.loc[year]
        # sort distributors by this year's movie count (descending)
        sorted_by_value = row.sort_values(ascending=False)
        bottom = 0
        for dist, value in sorted_by_value.items():
            if value <= 0:
                continue
            ax.bar(i, value, bottom=bottom, color=color_map.get(dist), width=width)
            ax.text(i, bottom + value / 2, f'{int(value)}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            bottom += value

    # Create legend using the stable color mapping ordered by overall totals
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color_map[dist], label=dist) for dist in distributors_ordered_by_total if dist in distributors]
    ax.legend(handles=legend_handles, title='Theatrical Distributor', bbox_to_anchor=(1, 1), loc='upper left')

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=0)
    plt.title('Number of Movies in the Top 100 Box Office by Theatrical Distributor Over Time (Top 8 Distributors)')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Movies Released')

    fig.tight_layout()
    # Save figure to output folder
    if save_fig:
        fig.savefig('../outputs/box_office_analysis/movie_count_per_distributor_overtime_top8.png', dpi=300, bbox_inches='tight')
    plt.show()

def worldwide_box_office_trend_by_genre(df, show=True, save_fig=False):
    # Group by Genre and sum Worldwide Box Office to get top genres
    genre_totals = df.groupby('Genre')['Worldwide Box Office'].sum().sort_values(ascending=False)
    top_genres = genre_totals.head(5).index
    # Filter data for top genres
    df_top = df[df['Genre'].isin(top_genres)]
    # Group by Release Year and Genre, sum Worldwide Box Office
    trend_df = df_top.groupby(['Release Year', 'Genre'])['Worldwide Box Office'].sum().reset_index()
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=trend_df, x='Release Year', y='Worldwide Box Office', hue='Genre', marker='o', ax=ax)
    ax.set_title('Worldwide Box Office Trend by Genre per Release Year (Top 5 Genres)')
    ax.set_xlabel('Release Year')
    ax.set_ylabel('Worldwide Box Office (Billions USD)')
    ax.legend(title='Genre')
    ax.grid(True)
    # Format y-axis to show in billions
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'${x/1e9:.1f}B'))
    # Save figure to output folder
    if save_fig:
        fig.savefig('../outputs/box_office_analysis/worldwide_box_office_trend_by_genre.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

def top_5_genres_percentage_per_year(df, show=True, save_fig=False):
    # Overall top 5 genres by total movie count
    overall_top_genres = df['Genre'].value_counts().head(5).index
    # Filter to these genres
    df_top_genres = df[df['Genre'].isin(overall_top_genres)]
    # Group by Release Year and Genre, count movies
    trend_count = df_top_genres.groupby(['Release Year', 'Genre']).size().reset_index(name='Movie Count')
    # Pivot to have years as index, genres as columns
    pivot_df = trend_count.pivot(index='Release Year', columns='Genre', values='Movie Count').fillna(0)
    # Calculate percentages
    pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    # Plot stacked bar
    ax = pivot_pct.plot(kind='bar', stacked=True, figsize=(12, 8))
    fig = ax.get_figure()
    plt.title('Percentage of Movies by Genre per Release Year (Top 5 Genres Overall)')
    plt.xlabel('Release Year')
    plt.ylabel('Percentage of Movies (%)')
    plt.legend(title='Genre', bbox_to_anchor=(1, 1), loc='upper left')
    # Add data labels inside the bars
    for i, (year, row) in enumerate(pivot_pct.iterrows()):
        cumulative = 0
        for genre in pivot_pct.columns:
            value = row[genre]
            if value > 0:
                ax.text(i, cumulative + value / 2, f'{value:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
                cumulative += value
    if save_fig:
        fig.savefig('../outputs/box_office_analysis/top_5_genres_percentage_per_year.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

def top_8_genres_percentage_per_year(df, show=True, save_fig=False):
    # Overall top 8 genres by total movie count
    overall_top_genres = df['Genre'].value_counts().head(8).index
    # Filter to these genres
    df_top_genres = df[df['Genre'].isin(overall_top_genres)]
    # Group by Release Year and Genre, count movies
    trend_count = df_top_genres.groupby(['Release Year', 'Genre']).size().reset_index(name='Movie Count')
    # Pivot to have years as index, genres as columns
    pivot_df = trend_count.pivot(index='Release Year', columns='Genre', values='Movie Count').fillna(0)
    # Calculate percentages
    pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    # Plot stacked bar
    ax = pivot_pct.plot(kind='bar', stacked=True, figsize=(14, 10))
    fig = ax.get_figure()
    plt.title('Percentage of Movies by Genre per Release Year (Top 8 Genres Overall)')
    plt.xlabel('Release Year')
    plt.ylabel('Percentage of Movies (%)')
    plt.legend(title='Genre', bbox_to_anchor=(1, 1), loc='upper left')
    # Add data labels inside the bars
    for i, (year, row) in enumerate(pivot_pct.iterrows()):
        cumulative = 0
        for genre in pivot_pct.columns:
            value = row[genre]
            if value > 0:
                ax.text(i, cumulative + value / 2, f'{value:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
                cumulative += value
    if save_fig:
        fig.savefig('../outputs/box_office_analysis/top_8_genres_percentage_per_year.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
        

### OSCARS

# Winners per year, user selected
def oscars_winners_table(df, year, save_fig=False): # table style
    styled = (
        df.style
          .hide(axis="index")
          .set_caption(f"Oscar Winners for the Year {year}")
          .format({"TMDb Average Vote": "{:.2f}"})
          .set_table_styles([
              # Title
              {"selector": "caption",
                  "props": [("font-size", "18px"),("font-weight", "bold"),("text-align", "left"),("margin-bottom", "10px")]},
              # Force header alignment to match body
              {"selector": "th",
                  "props": [("vertical-align", "top"),("text-align", "right"),("padding", "6px 10px"),("font-size", "13px")]},
              # Force cell alignment
              {"selector": "td",
                  "props": [("vertical-align", "top"),("padding", "6px 10px"),("font-size", "12px"),("line-height", "1.3")]},
              # Make column width calculations predictable
              {"selector": "table",
                  "props": [("table-layout", "fixed"),("width", "100%")]},])
          .set_properties(subset=["Production Companies"],
              **{"white-space": "normal","word-wrap": "break-word","max-width": "500px"}
          ))
    if save_fig:
            os.makedirs('../outputs/awards_analysis_oscars', exist_ok=True)
            dfi.export(styled,f'../outputs/awards_analysis_oscars/oscars_winners_{year}_table.png', table_conversion='matplotlib')

    return styled
    
def oscars_winners(df, year): # table entries & columns
    df = df[df['_merge'] == 'both']
    winners = (
        df.query("Year == @year and Winner == True")
          .loc[:, [
              "CanonicalCategory","Film","Name","production_companies","genres","vote_average"]
               ].rename(columns={"CanonicalCategory": "Category","Name": "Winner",
              "production_companies": "Production Companies","genres":"Genre","vote_average":"TMDb Average Vote"
          })
          .sort_values("Category").reset_index(drop=True))
    winners["TMDb Average Vote"] = winners["TMDb Average Vote"].round(2)
    return winners

# function to filter oscars df by wins and add columns for analysis: genres, companies & decades
def prep_oscars_wins(df):
    df = df[df['_merge'] == 'both'] # keep only matched Oscar–TMDb rows
    df = df[df['Winner'] == True] # keep only winners
    # Genres
    genres_of_interest = ['action', 'science fiction', 'comedy', 'drama', 'adventure', 'thriller', 'fantasy', 'romance', 'crime', 'animation', 'family', 'horror', 'western']
    # Create dummy columns for Genres (1 if genre in row, 0 otherwise; case-insensitive)
    for genre in genres_of_interest:
        df[genre] = df['genres'].str.lower().str.contains(genre.lower(), na=False).astype(int)
    # Production Companies
    production_companies = ['warner bros','metro-goldwyn-mayer', '20th century', 'marvel studios', 'paramount', 'universal','sony','disney','columbia pictures','legendary pictures','pixar','lionsgate','dreamworks','miramax','new line cinema']
    # Create dummy columns per company (1 if company in row, 0 otherwise; case-insensitive)
    for company in production_companies:
        df[company] = df['production_companies'].str.lower().str.contains(company.lower(), na=False).astype(int)
    # Add decade column
    df["Decade"] = (df["Year"] // 10) * 10
    df["Decade"] = df["Decade"].astype(int).astype(str) + "–" + (df["Decade"].astype(int) + 9).astype(str)
    df.loc[df["Year"] >= 2020, "Decade"] = "2020–2024"
    return df

def total_oscar_wins_per_genre(df, save_fig=False):
    df = prep_oscars_wins(df)
    genres_of_interest = ['action', 'science fiction', 'comedy', 'drama', 'adventure', 'thriller', 'fantasy', 'romance', 'crime', 'animation', 'family', 'horror', 'western']
    # Melt Genre data to long format for aggregation (one row per movie-genre pair)
    df_melted = df.melt(id_vars=['Film'], value_vars=genres_of_interest, var_name='genre', value_name='present')
    df_melted = df_melted[df_melted['present'] == 1].drop(columns='present')  # Keep only present genres
    # Count wins per genre
    genre_counts = (df_melted.groupby('genre')).size().sort_values(ascending=True)
    fig,ax = plt.subplots(figsize=(12,8))
    y = range(len(genre_counts))
    bars = ax.barh(y, genre_counts.values, color=sns.color_palette('plasma', len(genre_counts)))
    ax.set_title("Oscar Wins per Genre from 1980 to 2024")
    ax.set_xlabel("Number of Oscar Wins")
    ax.set_ylabel("Genre")
    ax.set_yticks(y)
    ax.set_yticklabels(genre_counts.index)
    for bar, count in zip(bars, genre_counts.values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,  f'{count:.0f}', va='center', fontsize=9)
    ax.grid(axis='x',linestyle='--',alpha=0.4)
    fig.tight_layout()
    if save_fig:
        fig.savefig('../outputs/awards_analysis_oscars/total_oscar_wins_per_genre.png', dpi=300, bbox_inches='tight')
    plt.show()

def oscar_wins_top_genres_per_decade(df,top_n=4, save_fig=False):
    df = prep_oscars_wins(df)
    genres_of_interest = ['action', 'science fiction', 'comedy', 'drama', 'adventure', 'thriller', 'fantasy', 'romance', 'crime', 'animation', 'family', 'horror', 'western']
    df_long = df.melt(
        id_vars=["Film", "Decade"],
        value_vars=genres_of_interest,
        var_name="genre",
        value_name="present")
    df_long = df_long[df_long["present"] == 1]
    # Count wins per decade & genre
    counts = (df_long.groupby(["Decade", "genre"]).size().reset_index(name="wins"))
    # Rank genres within each decade
    counts["rank"] = (counts.groupby("Decade")["wins"].rank(method="first", ascending=False))
    # Keep only top N genres per decade
    counts_top = counts[counts["rank"] <= top_n]
    # Pivot for stacked bar plot
    pivot = (counts_top.pivot(index="Decade", columns="genre", values="wins").fillna(0))
    # Ensure chronological order
    pivot = pivot.sort_index()
    # Plot
    fig, ax = plt.subplots(figsize=(13, 7))
    left = np.zeros(len(pivot))
    colors = sns.color_palette("Paired", pivot.shape[1])
    for genre, color in zip(pivot.columns, colors):
        bars = ax.barh(pivot.index,pivot[genre],left=left,label=genre,color=color)
        # Add value labels inside segments
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax.text(bar.get_x() + width / 2,bar.get_y() + bar.get_height() / 2,f"{int(width)}",
                    ha="center",va="center",fontsize=9,color="white")
        left += pivot[genre].values
    ax.set_title(f"Oscar Wins by Genre per Decade (Top {top_n} per Decade)")
    ax.set_xlabel("Number of Oscar Wins")
    ax.set_ylabel("Year Period")
    ax.legend(title="Genre", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis='x',linestyle='--',alpha=0.4)
    plt.tight_layout()
    if save_fig:
        fig.savefig(f'../outputs/awards_analysis_oscars/oscar_wins_top{top_n}_genres_per_decade.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def total_oscar_wins_per_production_company(df, save_fig=False):
    df = prep_oscars_wins(df)
    production_companies = ['warner bros','metro-goldwyn-mayer', '20th century', 'marvel studios', 'paramount', 'universal','sony','disney','columbia pictures','legendary pictures','pixar','lionsgate','dreamworks','miramax','new line cinema']

    # Melt company data to long format for aggregation (one row per movie-company pair)
    df_melted = df.melt(id_vars=['Film'], value_vars=production_companies, var_name='production_company', value_name='present')
    df_melted = df_melted[df_melted['present'] == 1].drop(columns='present')  # Keep only present genres
    # Count wins per production company
    company_counts = (df_melted.groupby('production_company')).size().sort_values(ascending=True)
    fig,ax = plt.subplots(figsize=(12,8))
    y = range(len(company_counts))
    bars = ax.barh(y, company_counts.values, color=sns.color_palette('viridis_r', len(company_counts)))
    ax.set_title("Oscar Wins per Production Company from 1980 to 2024")
    ax.set_xlabel("Number of Oscar Wins")
    ax.set_ylabel("Production Company")
    ax.set_yticks(y)
    ax.set_yticklabels(company_counts.index)
    for bar, count in zip(bars, company_counts.values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,  f'{count:.0f}', va='center', fontsize=9)
    ax.grid(axis='x',linestyle='--',alpha=0.4)
    fig.tight_layout()
    if save_fig:
        fig.savefig(f'../outputs/awards_analysis_oscars/total_oscar_wins_per_production_company.png', dpi=300, bbox_inches='tight')
    plt.show()

# Production Companies with the most Oscar wins per decade
def oscar_wins_top_companies_per_decade(df,top_n=4, save_fig=False):
    df = prep_oscars_wins(df)
    production_companies = ['warner bros','metro-goldwyn-mayer', '20th century', 'marvel studios', 'paramount', 'universal','sony','disney','columbia pictures','legendary pictures','pixar','lionsgate','dreamworks','miramax','new line cinema']
    df_long = df.melt(id_vars=["Film", "Decade"],
        value_vars=production_companies,
        var_name="production_company",
        value_name="present")
    df_long = df_long[df_long["present"] == 1]
    # Count wins per decade & companies
    counts = (df_long.groupby(["Decade", "production_company"]).size().reset_index(name="wins"))
    # Rank companies within each decade
    counts["rank"] = (counts.groupby("Decade")["wins"].rank(method="first", ascending=False))
    # Keep only top N genres per decade
    counts_top = counts[counts["rank"] <= top_n]
    # Pivot for stacked bar plot
    pivot = (counts_top.pivot(index="Decade", columns="production_company", values="wins").fillna(0))
    # Ensure chronological order
    pivot = pivot.sort_index()
    # Plot
    fig, ax = plt.subplots(figsize=(13, 7))
    left = np.zeros(len(pivot))
    colors = sns.color_palette("muted", pivot.shape[1])
    for company, color in zip(pivot.columns, colors):
        bars = ax.barh(pivot.index,pivot[company],left=left,label=company,color=color)
        # Add value labels inside segments
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax.text(bar.get_x() + width / 2,bar.get_y() + bar.get_height() / 2,f"{int(width)}",
                    ha="center",va="center",fontsize=9,color="white")
        left += pivot[company].values
    ax.set_title(f"Oscar Wins by Production Company per Decade (Top {top_n} per Decade)")
    ax.set_xlabel("Number of Oscar Wins")
    ax.set_ylabel("Year Period")
    ax.legend(title="Production Company", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis='x',linestyle='--',alpha=0.4)
    plt.tight_layout()
    if save_fig:
        fig.savefig(f'../outputs/awards_analysis_oscars/oscar_wins_top{top_n}_production_companies_per_decade.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### NETFLIX

def netflix_spend(df, save_fig=False):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='Year', y='Netflix Content Spending', color='#E50914', ax=ax)
    ax.set_title('Netflix Content Spend Over Time', fontweight='semibold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Netflix Content Spending')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x/1e9:.0f}B'))
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    if save_fig:
        fig.savefig(f'../outputs/streaming_analysis_netflix/netflix_spend.png', dpi=300, bbox_inches='tight')
    plt.show()

def netflix_rev_subs(df, save_fig=False):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    sns.lineplot(x='Year', y='Netflix Revenue', data=df, ax=ax1, color='blue', label='Revenue', legend=False)
    sns.lineplot(x='Year', y='Netflix Subscribers', data=df, ax=ax2, color='#E50914', label='Subscribers', legend=False)
    years = sorted(df['Year'].dropna().unique())
    ax1.set_title('Netflix Revenue and Subscribers Over Time', fontweight='semibold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Netflix Revenue', color='blue')
    ax2.set_ylabel('Netflix Subscribers', color='#E50914')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x/1e9:.0f}B'))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.0f}M'))
    ax1.grid(False)
    ax1.grid(axis='x', color='lightgrey', linestyle='--', linewidth=0.5)
    ax1.set_xticks(years)
    if years:
        ax1.set_xlim(years[0] - 0.5, years[-1] + 0.5)

    fig.tight_layout()
    if save_fig:
        fig.savefig(f'../outputs/streaming_analysis_netflix/netflix_revenue_vs_subscribers.png', dpi=300, bbox_inches='tight')
    plt.show()

def netflix_top_movies(df, save_fig=False):
    df_movies = df[df['Type'] == 'Movie']
    df_top_movies_by_hours = df_movies.sort_values(by='Total Hours Viewed', ascending=False).head(10)
    df_top_movies_by_views = df_movies.sort_values(by='Total Views', ascending=False).head(10)
    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(15,10))
    sns.barplot(data=df_top_movies_by_hours, x='Total Hours Viewed', y='Title Name', ax=axes[0],color='orange')
    axes[0].set_title('Top 10 Movies by Total Hours Viewed', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Movie Title', fontsize=12)
    axes[0].set_xlabel('Total Hours Viewed', fontsize=12)
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.0f}M'))
    # Add data labels inside the bars
    for p in axes[0].patches:
        width = p.get_width()
        axes[0].text(width - width * 0.05, p.get_y() + p.get_height() / 2, f'{width:,.0f}', ha='right', va='center', color='white', fontsize=9, fontweight='bold')

    sns.barplot(data=df_top_movies_by_views, x='Total Views', y='Title Name', ax=axes[1], color='#E50914')

    axes[1].set_title('Top 10 Movies by Total Views', fontsize=14, fontweight='bold') # Choose a Netflix font ??
    axes[1].set_ylabel('Movie Title', fontsize=12)
    axes[1].set_xlabel('Total Views', fontsize=12)
    # Add data labels inside the bars
    for p in axes[1].patches:
        width = p.get_width()
        axes[1].text(width - width * 0.05, p.get_y() + p.get_height() / 2, f'{width:,.0f}', ha='right', va='center', color='white', fontsize=9, fontweight='bold')
    plt.tight_layout()
    if save_fig:
        fig.savefig(f'../outputs/streaming_analysis_netflix/netflix_top_movies.png', dpi=300, bbox_inches='tight')
    plt.show()


def netflix_top_shows(df, save_fig=False):
    df_shows = df[df['Type'] == 'TV']
    df_top_shows_by_hours = df_shows.sort_values(by='Total Hours Viewed', ascending=False).head(10)
    df_top_shows_by_views = df_shows.sort_values(by='Total Views', ascending=False).head(10)

    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(15,10))
    sns.barplot(data=df_top_shows_by_hours, x='Total Hours Viewed', y='Title Name', ax=axes[0],color='#F5612C')

    axes[0].set_title('Top 10 TV Shows by Total Hours Viewed', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('TV Show Title', fontsize=12)
    axes[0].set_xlabel('Total Hours Viewed', fontsize=12)
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.0f}M'))
    # Add data labels inside the bars
    for p in axes[0].patches:
        width = p.get_width()
        axes[0].text(width - width * 0.05, p.get_y() + p.get_height() / 2, f'{width:,.0f}', ha='right', va='center', color='white', fontsize=9, fontweight='bold')

    sns.barplot(data=df_top_shows_by_views, x='Total Views', y='Title Name', ax=axes[1], color='#DE4991')

    axes[1].set_title('Top 10 TV Shows by Total Views', fontsize=14, fontweight='bold') # Choose a Netflix font ??
    axes[1].set_ylabel('TV Show Title', fontsize=12)
    axes[1].set_xlabel('Total Views', fontsize=12)
    # Add data labels inside the bars
    for p in axes[1].patches:
        width = p.get_width()
        axes[1].text(width - width * 0.05, p.get_y() + p.get_height() / 2, f'{width:,.0f}', ha='right', va='center', color='white', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()


########## END OF ANALYSIS FUNCTIONS ##########

if __name__ == "__main__":
    
    pass
