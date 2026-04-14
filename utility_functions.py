#!/usr/bin/env python3
"""
Tactical Analytics Utilities for Georgia Tech MSA Practicum.
This file contains consolidated functions for data loading, feature engineering, and visualization.
"""

from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

statsbomb_directory = Path("../data/Statsbomb")

def load_events():
    # Load Statsbomb events data from data directory
    events = pl.read_parquet(statsbomb_directory / "events.parquet")
    return events

def load_matches():
    return pl.read_parquet(statsbomb_directory / "matches.parquet")

def extract_tactical_dna(events_df):
    """
    Groups raw event data into match-level tactical features.
    """
    events_df = events_df.with_columns(
        is_risky_pass = pl.any_horizontal(
            pl.col("pass_through_ball").fill_null(False),
            pl.col("pass_cross").fill_null(False),
            pl.col("pass_switch").fill_null(False),
            pl.col("pass_cut_back").fill_null(False)
        )
    )

    events_df = events_df.with_columns(
        is_successful_pass = (pl.col("pass_outcome").is_null())
        )
    
    # Aggregation Logic and feature engineering
    style_matrix = (
        events_df.group_by(["match_id", "team"])
        .agg([
            # Passing Features
            (pl.col("is_successful_pass").filter(pl.col("type") == "Pass").sum() / 
            pl.col("type").filter(pl.col("type") == "Pass").count() * 100).alias("completion_rate"),
            
            pl.col("pass_length").mean().alias("avg_pass_length"),
            
            (pl.col("is_risky_pass").filter(pl.col("type") == "Pass").sum() / 
            pl.col("type").filter(pl.col("type") == "Pass").count() * 100).alias("risky_pass_rate"),
            
            (pl.col("pass_height").filter(pl.col("pass_height") == "High Pass").count() / 
            pl.col("type").filter(pl.col("type") == "Pass").count() * 100).alias("high_pass_rate"),
            (pl.col("pass_end_location_x") - pl.col("location_x")).mean().alias("avg_pass_verticaldistance"),

            # Possession/Dribbling Features
            (pl.col("carry_end_location_x") - pl.col("location_x")).mean().alias("avg_carry_distance"),
            
            (pl.col("dribble_outcome").filter(pl.col("dribble_outcome") == "Complete").count() / 
            pl.col("type").filter(pl.col("type") == "Dribble").count() * 100).alias("dribble_success_rate"),
            
            # Shooting Features
            pl.col("shot_statsbomb_xg").mean().alias("avg_shot_xg"),
            
            (((pl.when(pl.col("location_x") > 60).then(120).otherwise(0) - pl.col("location_x")).pow(2) + 
                (40 - pl.col("location_y")).pow(2)).sqrt()
            ).filter(pl.col("type") == "Shot").mean().alias("avg_shot_distance"),
            
            (pl.col("shot_follows_dribble").filter(pl.col("type") == "Shot").sum() / 
            pl.col("type").filter(pl.col("type") == "Shot").count() * 100).alias("solo_shot_rate"), 
            
            (pl.col("pass_through_ball").filter(pl.col("pass_assisted_shot_id").is_not_null()).sum() / 
            pl.col("type").filter(pl.col("type") == "Shot").count() * 100).alias("through_ball_shot_rate"),
            
            (pl.col("shot_statsbomb_xg").filter((pl.col("type") == "Shot") & (pl.col("shot_statsbomb_xg") > 0.3)).count() / 
            pl.col("type").filter(pl.col("type") == "Shot").count() * 100).alias("big_chance_shot_rate")
        ])
        .drop_nulls()
        .sort(["match_id", "team"])
    )

    return style_matrix

def prepare_style_matrix(df, features):
    """
    Standardizes tactical features and prepares the matrix for clustering.
    
    Args:
        df (pl.DataFrame): The raw match style statistics.
        features (list): The list of features to include.
        
    Returns:
        tuple: (style_matrix_raw, style_matrix_scaled, style_values_scaled)
    """
    # Create a clean matrix with only relevant features
    style_matrix_raw = df.select(["match_id", "team"] + features)

    # Standardize features (Mean=0, Variance=1)
    scaler = StandardScaler()
    style_values_scaled = scaler.fit_transform(style_matrix_raw.select(features).to_pandas())

    # Create a scaled DataFrame for easier visualization/analysis
    style_matrix_scaled = pd.DataFrame(
        style_values_scaled, 
        columns=features
    )
    
    # Re-attach identifiers
    style_matrix_scaled[['match_id', 'team']] = style_matrix_raw.select(["match_id", "team"]).to_pandas()
    
    return style_matrix_raw, style_matrix_scaled, style_values_scaled

def plot_tactical_radar(data, features, clean_labels, cluster_map=None, title="Tactical Play Style Analysis"):
    """
    A radar chart function for any number of clusters.

    Args:
        data (pd.DataFrame): The cluster profiles (mean values).
        features (list): The raw column names used in the data.
        clean_labels (list): The intuitive names for the radar axes.
        cluster_map (dict): Optional mapping of {cluster_id: "Custom Name"}.
        title (str): Chart title.
    """
    num_vars = len(features)
    num_clusters = len(data)
    
    # Create angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] 

    # Setup Figure and Axis
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Shrink the plot area to 70% to give 30% space for external labels
    ax.set_position([0.15, 0.15, 0.7, 0.7]) 

    # Generate a dynamic color palette based on the number of clusters
    colors = plt.colormaps['tab10'](range(num_clusters))

    # Iterate through each cluster in the data
    for i, (idx, row) in enumerate(data.iterrows()):
        # Get custom name if provided, otherwise default to Cluster #
        c_id = int(row['cluster'])
        label_text = cluster_map.get(c_id, f"Cluster {c_id}") if cluster_map else f"Cluster {c_id}"
        
        values = row[features].values.flatten().tolist()
        values += values[:1] 
        
        # Plot the line and fill
        ax.plot(angles, values, color=colors[i], linewidth=3, label=label_text)
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    # Aesthetics
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels and handle alignment to prevent bleeding
    ax.set_thetagrids(np.degrees(angles[:-1]), clean_labels, fontsize=10)
    
    for label, angle in zip(ax.get_xticklabels(), angles):
        if np.isclose(angle, 0) or np.isclose(angle, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    # Fixed scale for Z-scores (StandardScaler results)
    ax.set_ylim(-2.0, 2.0) 
    ax.set_yticklabels([]) # Hide radial numbers for a cleaner look

    plt.title(title, size=16, pad=60)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, frameon=True)
    
    plt.show()

def plot_tactical_pca(df, features, cluster_col='cluster', cluster_map=None):
    """
    Performs PCA on tactical features and visualizes cluster separation.
    
    Args:
        df (pl.DataFrame): The style matrix containing raw features and cluster labels.
        features (list): The list of features to use for dimensionality reduction.
        cluster_col (str): The column name for cluster assignments.
        cluster_map (dict): Optional mapping for legend (e.g., {0: "Chaos"}).
    """
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Scale and Transform
    # Standardizing is mandatory to ensure PCA isn't biased by feature units
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select(features).to_pandas())

    pca = PCA(n_components=2, random_state=21)
    pca_results = pca.fit_transform(scaled_data)

    # Prepare Plotting DataFrame
    pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
    
    # Map cluster IDs to Names if map is provided
    if cluster_map:
        pca_df['Tactical Style'] = df[cluster_col].map_elements(lambda x: cluster_map.get(x, x)).to_list()
        hue_col = 'Tactical Style'
    else:
        pca_df['Cluster'] = df[cluster_col].to_list()
        hue_col = 'Cluster'

    # Visualization
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=pca_df,
        x='PC1', y='PC2', 
        hue=hue_col, 
        palette='tab10', 
        s=100, 
        alpha=0.7,
        edgecolor='white'
    )

    # Annotate with explained variance
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    
    plt.xlabel(f"PC1 ({var_pc1:.1f}% Variance Explained)")
    plt.ylabel(f"PC2 ({var_pc2:.1f}% Variance Explained)")
    plt.title("PCA Dimensionality Reduction: Tactical Archetype Separation", fontsize=14, pad=20)
    
    plt.legend(title="Tactical Style", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    return pca 