import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import warnings
import requests
import zipfile
import tempfile
import shutil
import subprocess
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from datetime import datetime
import pickle
import random
import json

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def check_models_freshness():
    """Check if models need updating based on data freshness"""
    try:
        config_file = "data_update_config.json"
        models_dir = "trained_models"
        
        if not os.path.exists(config_file):
            return False, "No update config found"
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        last_update = config.get('last_update')
        if not last_update:
            return False, "No update history found"
        
        # Check if models exist
        required_models = ['score_model.pkl', 'outcome_model.pkl']
        for model_file in required_models:
            if not os.path.exists(os.path.join(models_dir, model_file)):
                return False, f"Missing model: {model_file}"
        
        # Check model age vs data age
        last_update_time = datetime.fromisoformat(last_update)
        model_files = [os.path.join(models_dir, f) for f in required_models]
        
        for model_path in model_files:
            model_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            if last_update_time > model_time:
                return False, "Models are older than latest data"
        
        # Check if data is recent (within last 7 days)
        days_since_update = (datetime.now() - last_update_time).days
        if days_since_update > 7:
            return False, f"Data is {days_since_update} days old"
        
        return True, "Models are up to date"
        
    except Exception as e:
        return False, f"Error checking model freshness: {str(e)}"

def install_package(package):
    """Install required packages if not already installed"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

# List of required packages
required_packages = [
    "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "xgboost", "requests"
]

# Silently check and install packages
for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        install_package(package)

def download_cricsheet_data(url, extract_to="cricsheet_data"):
    """Download and extract cricket data from Cricsheet"""
    # Create directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    # Download the zip file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_file_path = tmp_file.name
    
    # Extract the zip file
    with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Clean up temporary file
    os.unlink(tmp_file_path)
    
    # Count extracted files
    csv_files = glob(os.path.join(extract_to, "*.csv"))
    
    return extract_to, csv_files

def parse_cricket_csv(file_path):
    """Parse a cricket CSV file and extract match info and ball-by-ball data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    match_info = {}
    players = {'team1': [], 'team2': []}
    ball_data = []
    teams = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(',')
        
        if parts[0] == 'info':
            if parts[1] == 'team':
                teams.append(parts[2])
            elif parts[1] == 'player':
                team = parts[2]
                player = parts[3]
                if team == teams[0] if teams else None:
                    players['team1'].append(player)
                elif team == teams[1] if len(teams) > 1 else None:
                    players['team2'].append(player)
            else:
                match_info[parts[1]] = ','.join(parts[2:]) if len(parts) > 2 else ''
        
        elif parts[0] == 'ball':
            ball_info = {
                'innings': int(parts[1]),
                'over_ball': parts[2],
                'batting_team': parts[3],
                'striker': parts[4],
                'non_striker': parts[5],
                'bowler': parts[6],
                'runs_batter': int(parts[7]) if parts[7] else 0,
                'runs_extras': int(parts[8]) if parts[8] else 0,
                'wides': int(parts[9]) if parts[9] else 0,
                'noballs': int(parts[10]) if parts[10] else 0,
                'byes': int(parts[11]) if parts[11] else 0,
                'legbyes': int(parts[12]) if parts[12] else 0,
                'penalty': int(parts[13]) if parts[13] else 0,
                'wicket_type': parts[14] if len(parts) > 14 and parts[14] else None,
                'player_dismissed': parts[15] if len(parts) > 15 and parts[15] else None
            }
            
            # Parse over and ball
            over_parts = parts[2].split('.')
            ball_info['over'] = int(over_parts[0])
            ball_info['ball'] = int(over_parts[1])
            ball_info['total_runs'] = ball_info['runs_batter'] + ball_info['runs_extras']
            
            ball_data.append(ball_info)
    
    # Add team names to match info
    if len(teams) >= 2:
        match_info['team1'] = teams[0]
        match_info['team2'] = teams[1]
    
    return match_info, players, pd.DataFrame(ball_data)

def create_features(ball_df, match_info):
    """Create features for machine learning models"""
    df = ball_df.copy()
    
    # Basic features
    df['ball_number'] = df.groupby('innings').cumcount() + 1
    df['over_progress'] = df['ball'] / 6
    df['match_progress'] = df['ball_number'] / df.groupby('innings')['ball_number'].transform('max')
    
    # Running totals
    df['cumulative_runs'] = df.groupby('innings')['total_runs'].cumsum()
    df['cumulative_wickets'] = df.groupby('innings')['wicket_type'].transform(lambda x: x.notna().cumsum())
    
    # Recent form (last 6 balls)
    df['recent_runs'] = df.groupby('innings')['total_runs'].rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
    df['recent_dot_balls'] = df.groupby('innings')['runs_batter'].rolling(6, min_periods=1).apply(lambda x: (x == 0).sum()).reset_index(level=0, drop=True)
    
    # Current run rate
    df['current_rr'] = df['cumulative_runs'] / (df['ball_number'] / 6)
    df['current_rr'] = df['current_rr'].fillna(0)
    
    # Required run rate (for second innings)
    if len(df['innings'].unique()) > 1:
        first_innings_total = df[df['innings'] == 1]['cumulative_runs'].max()
        df['target'] = first_innings_total + 1
        df['runs_needed'] = np.where(df['innings'] > 1, 
                                   df['target'] - df['cumulative_runs'], 
                                   0)
        total_balls = df[df['innings'] == 1]['ball_number'].max()
        df['balls_remaining'] = np.where(df['innings'] > 1,
                                       total_balls - df['ball_number'],
                                       0)
        df['required_rr'] = np.where(df['balls_remaining'] > 0,
                                   (df['runs_needed'] * 6) / df['balls_remaining'],
                                   0)
    else:
        df['target'] = 0
        df['runs_needed'] = 0
        df['balls_remaining'] = 0
        df['required_rr'] = 0
    
    # Pressure indicators
    df['wickets_in_hand'] = 10 - df['cumulative_wickets']
    df['pressure_index'] = df['required_rr'] / (df['wickets_in_hand'] + 1)
    
    # Encode categorical variables
    le_batsman = LabelEncoder()
    le_bowler = LabelEncoder()
    le_team = LabelEncoder()
    
    df['striker_encoded'] = le_batsman.fit_transform(df['striker'])
    df['bowler_encoded'] = le_bowler.fit_transform(df['bowler'])
    df['batting_team_encoded'] = le_team.fit_transform(df['batting_team'])
    
    return df, {'batsman_encoder': le_batsman, 'bowler_encoder': le_bowler, 'team_encoder': le_team}

def build_score_prediction_model(combined_balls_data):
    """Build a model to predict final score based on current match state"""
    # Prepare data for score prediction
    score_data = []
    
    # Handle single match data (no match_id column)
    if 'match_id' not in combined_balls_data.columns:
        # Use the entire dataset as one match
        match_data = combined_balls_data
        for innings in match_data['innings'].unique():
            innings_data = match_data[match_data['innings'] == innings]
            final_score = innings_data['cumulative_runs'].max()
            
            # Create training samples at different points in the innings
            for i in range(10, len(innings_data), 6):  # Every over after the 10th ball
                if i < len(innings_data):
                    current_state = innings_data.iloc[i]
                    
                    sample = {
                        'overs_completed': current_state['over'],
                        'current_score': current_state['cumulative_runs'],
                        'wickets_fallen': current_state['cumulative_wickets'],
                        'current_rr': current_state['current_rr'],
                        'recent_runs': current_state['recent_runs'],
                        'wickets_in_hand': current_state['wickets_in_hand'],
                        'striker_encoded': current_state['striker_encoded'],
                        'bowler_encoded': current_state['bowler_encoded'],
                        'final_score': final_score
                    }
                    score_data.append(sample)
    else:
        # Handle multiple matches data
        for match_id in combined_balls_data['match_id'].unique():
            match_data = combined_balls_data[combined_balls_data['match_id'] == match_id]
            
            for innings in match_data['innings'].unique():
                innings_data = match_data[match_data['innings'] == innings]
                final_score = innings_data['cumulative_runs'].max()
                
                # Create training samples at different points in the innings
                for i in range(10, len(innings_data), 6):  # Every over after the 10th ball
                    if i < len(innings_data):
                        current_state = innings_data.iloc[i]
                        
                        sample = {
                            'overs_completed': current_state['over'],
                            'current_score': current_state['cumulative_runs'],
                            'wickets_fallen': current_state['cumulative_wickets'],
                            'current_rr': current_state['current_rr'],
                            'recent_runs': current_state['recent_runs'],
                            'wickets_in_hand': current_state['wickets_in_hand'],
                            'striker_encoded': current_state['striker_encoded'],
                            'bowler_encoded': current_state['bowler_encoded'],
                            'final_score': final_score
                        }
                        score_data.append(sample)
    
    if len(score_data) == 0:
        return None, None, None
    
    score_df = pd.DataFrame(score_data)
    
    # Features for prediction
    feature_columns = ['overs_completed', 'current_score', 'wickets_fallen', 
                      'current_rr', 'recent_runs', 'wickets_in_hand',
                      'striker_encoded', 'bowler_encoded']
    
    # Check if all required features exist
    missing_features = [f for f in feature_columns if f not in score_df.columns]
    if missing_features:
        return None, None, None
    
    X = score_df[feature_columns]
    y = score_df['final_score']
    
    # Check if we have enough data
    if len(X) < 10:
        return None, None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            results[name] = {'model': model, 'mse': mse, 'rmse': np.sqrt(mse)}
        except Exception as e:
            continue
    
    if not results:
        return None, None, None
    
    # Return the best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
    best_model = results[best_model_name]['model']
    
    return best_model, feature_columns, results

def build_match_outcome_model(all_matches_data, combined_balls_data):
    """Build a model to predict match outcome"""
    match_features = []
    
    for match_info in all_matches_data:
        match_id = match_info['match_id']
        match_balls = combined_balls_data[combined_balls_data['match_id'] == match_id]
        
        if len(match_balls) == 0:
            continue
            
        # Extract match-level features
        team1 = match_info.get('team1', 'Unknown')
        team2 = match_info.get('team2', 'Unknown')
        winner = match_info.get('winner', 'Unknown')
        
        # Calculate team statistics
        team1_balls = match_balls[match_balls['batting_team'] == team1]
        team2_balls = match_balls[match_balls['batting_team'] == team2]
        
        if len(team1_balls) > 0 and len(team2_balls) > 0:
            team1_score = team1_balls['cumulative_runs'].max()
            team2_score = team2_balls['cumulative_runs'].max()
            
            team1_wickets = team1_balls['cumulative_wickets'].max()
            team2_wickets = team2_balls['cumulative_wickets'].max()
            
            team1_sr = team1_balls['current_rr'].mean()
            team2_sr = team2_balls['current_rr'].mean()
            
            # Toss impact
            toss_winner = match_info.get('toss_winner', 'Unknown')
            toss_decision = match_info.get('toss_decision', 'Unknown')
            
            match_features.append({
                'team1_score': team1_score,
                'team2_score': team2_score,
                'team1_wickets': team1_wickets,
                'team2_wickets': team2_wickets,
                'team1_strike_rate': team1_sr,
                'team2_strike_rate': team2_sr,
                'score_difference': abs(team1_score - team2_score),
                'toss_won_by_team1': 1 if toss_winner == team1 else 0,
                'toss_bat_first': 1 if toss_decision == 'bat' else 0,
                'winner_team1': 1 if winner == team1 else 0
            })
    
    if len(match_features) == 0:
        return None, None
    
    outcome_df = pd.DataFrame(match_features)
    
    # Features for prediction
    feature_cols = ['team1_score', 'team2_score', 'team1_wickets', 'team2_wickets',
                   'team1_strike_rate', 'team2_strike_rate', 'score_difference',
                   'toss_won_by_team1', 'toss_bat_first']
    
    X = outcome_df[feature_cols].fillna(0)
    y = outcome_df['winner_team1']
    
    if len(X) < 4:  # Need at least 4 samples for train/test split
        return None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, feature_cols

class CricketMatchPredictor:
    """Comprehensive cricket match prediction system"""
    
    def __init__(self, score_model=None, wicket_model=None, outcome_model=None, 
                 score_features=None, wicket_features=None, outcome_features=None):
        self.score_model = score_model
        self.wicket_model = wicket_model  
        self.outcome_model = outcome_model
        self.score_features = score_features
        self.wicket_features = wicket_features
        self.outcome_features = outcome_features
        
        # Team strength ratings (can be updated based on recent performance)
        self.team_ratings = {
            'India': 85, 'Australia': 82, 'England': 80, 'New Zealand': 78,
            'South Africa': 76, 'Pakistan': 74, 'West Indies': 70, 
            'Sri Lanka': 68, 'Bangladesh': 65, 'Afghanistan': 62
        }
        
    def predict_match_winner(self, team1, team2, match_format='ODI', venue='neutral'):
        """Predict match winner based on team strengths and conditions"""
        print(f"🏏 MATCH PREDICTION: {team1} vs {team2}")
        print(f"📊 Format: {match_format} | Venue: {venue}")
        print("=" * 50)
        
        # Get team ratings
        team1_rating = self.team_ratings.get(team1, 60)
        team2_rating = self.team_ratings.get(team2, 60)
        
        # Adjust for home advantage
        if venue.lower() == team1.lower():
            team1_rating += 5
        elif venue.lower() == team2.lower():
            team2_rating += 5
            
        # Format adjustments
        if match_format == 'T20':
            # T20 is more unpredictable
            variance = 15
        elif match_format == 'ODI':
            variance = 10
        else:  # Test
            variance = 8
            
        # Calculate win probabilities
        rating_diff = team1_rating - team2_rating
        team1_prob = 0.5 + (rating_diff / (2 * variance))
        team1_prob = max(0.1, min(0.9, team1_prob))  # Keep between 10-90%
        team2_prob = 1 - team1_prob
        
        # Determine winner
        winner = team1 if team1_prob > team2_prob else team2
        confidence = max(team1_prob, team2_prob)
        
        print(f"🎯 PREDICTION RESULTS:")
        print(f"   {team1}: {team1_prob:.1%} chance to win")
        print(f"   {team2}: {team2_prob:.1%} chance to win")
        print(f"   Predicted Winner: {winner}")
        print(f"   Confidence Level: {confidence:.1%}")
        
        return {
            'winner': winner,
            'team1_probability': team1_prob,
            'team2_probability': team2_prob,
            'confidence': confidence
        }
    
    def predict_match_scores(self, team1, team2, match_format='ODI'):
        """Predict expected scores for both teams"""
        print(f"\n📊 SCORE PREDICTION: {team1} vs {team2}")
        print("=" * 40)
        
        # Base scores by format
        if match_format == 'T20':
            base_score = 160
            score_range = 40
        elif match_format == 'ODI':
            base_score = 280
            score_range = 70
        else:  # Test (first innings)
            base_score = 350
            score_range = 150
            
        # Adjust based on team strength
        team1_rating = self.team_ratings.get(team1, 60)
        team2_rating = self.team_ratings.get(team2, 60)
        
        # Calculate expected scores
        team1_score = base_score + ((team1_rating - 70) * 2) + random.randint(-score_range//2, score_range//2)
        team2_score = base_score + ((team2_rating - 70) * 2) + random.randint(-score_range//2, score_range//2)
        
        # Ensure realistic minimums
        min_score = {'T20': 120, 'ODI': 200, 'Test': 250}
        team1_score = max(min_score.get(match_format, 200), team1_score)
        team2_score = max(min_score.get(match_format, 200), team2_score)
        
        print(f"   Expected {team1} score: {team1_score}")
        print(f"   Expected {team2} score: {team2_score}")
        
        # Score ranges
        team1_range = (team1_score - 30, team1_score + 30)
        team2_range = (team2_score - 30, team2_score + 30)
        
        print(f"   {team1} score range: {team1_range[0]}-{team1_range[1]}")
        print(f"   {team2} score range: {team2_range[0]}-{team2_range[1]}")
        
        return {
            'team1_score': team1_score,
            'team2_score': team2_score,
            'team1_range': team1_range,
            'team2_range': team2_range
        }

def predict_cricket_match(team1, team2, match_format='ODI', venue='neutral', 
                          weather='clear', current_score=None, current_wickets=None, 
                          current_over=None, target=None):
    """
    🏏 ONE-STOP CRICKET MATCH PREDICTION FUNCTION
    
    Parameters:
    -----------
    team1, team2 : str
        Team names (e.g., 'India', 'Australia', 'England', etc.)
    match_format : str
        'T20', 'ODI', or 'Test'
    venue : str  
        'neutral', team1 name for home advantage, or team2 name
    weather : str
        'clear', 'overcast', 'sunny', 'windy', 'humid'
    current_score : int (optional)
        Current score for live match tracking
    current_wickets : int (optional)
        Current wickets fallen for live tracking  
    current_over : float (optional)
        Current over (e.g., 25.3) for live tracking
    target : int (optional)
        Target score if team is chasing
        
    Returns:
    --------
    Complete match prediction with all analysis
    """
    
    # Check model freshness
    models_fresh, freshness_msg = check_models_freshness()
    
    print("🏏" + "="*60 + "🏏")
    print(f"    CRICKET MATCH PREDICTION: {team1} vs {team2}")
    print("🏏" + "="*60 + "🏏")
    print(f"📊 Format: {match_format} | Venue: {venue} | Weather: {weather}")
    
    if not models_fresh:
        print(f"⚠️  Model Freshness Warning: {freshness_msg}")
        print("💡 Consider running data update for more accurate predictions")
    
    # Initialize team ratings
    team_ratings = {
        'India': 85, 'Australia': 82, 'England': 80, 'New Zealand': 78,
        'South Africa': 76, 'Pakistan': 74, 'West Indies': 70, 
        'Sri Lanka': 68, 'Bangladesh': 65, 'Afghanistan': 62,
        'Zimbabwe': 58, 'Ireland': 55, 'Netherlands': 52
    }
    
    team1_rating = team_ratings.get(team1, 60)
    team2_rating = team_ratings.get(team2, 60)
    
    # Venue adjustments
    if venue.lower() == team1.lower():
        team1_rating += 5
        venue_advantage = f"{team1} (Home)"
    elif venue.lower() == team2.lower():
        team2_rating += 5
        venue_advantage = f"{team2} (Home)"
    else:
        venue_advantage = "Neutral"
    
    # Weather adjustments
    weather_impact = {
        'overcast': {'batting': -10, 'bowling': 'Swing favorable'},
        'sunny': {'batting': +5, 'bowling': 'Batting friendly'},
        'windy': {'batting': -5, 'bowling': 'Spin difficult'},
        'humid': {'batting': -5, 'bowling': 'Reverse swing possible'},
        'clear': {'batting': 0, 'bowling': 'Balanced conditions'}
    }
    
    weather_adj = weather_impact.get(weather, {'batting': 0, 'bowling': 'Normal'})
    
    # 1. MATCH WINNER PREDICTION
    print(f"\n🎯 WINNER PREDICTION")
    print("=" * 25)
    
    # Calculate win probabilities
    variance = {'T20': 15, 'ODI': 10, 'Test': 8}[match_format]
    rating_diff = team1_rating - team2_rating
    team1_prob = 0.5 + (rating_diff / (2 * variance))
    team1_prob = max(0.15, min(0.85, team1_prob))
    team2_prob = 1 - team1_prob
    
    winner = team1 if team1_prob > team2_prob else team2
    confidence = max(team1_prob, team2_prob)
    
    print(f"🏆 Predicted Winner: {winner}")
    print(f"📊 {team1}: {team1_prob:.1%} | {team2}: {team2_prob:.1%}")
    print(f"🎯 Confidence: {confidence:.1%}")
    print(f"🏟️  Venue Advantage: {venue_advantage}")
    
    # 2. SCORE PREDICTION
    print(f"\n📊 SCORE PREDICTION")
    print("=" * 20)
    
    # Base scores by format
    base_scores = {'T20': 160, 'ODI': 280, 'Test': 350}
    score_ranges = {'T20': 40, 'ODI': 70, 'Test': 150}
    
    base_score = base_scores[match_format]
    score_range = score_ranges[match_format]
    
    # Calculate team scores
    team1_score = base_score + ((team1_rating - 70) * 2) + weather_adj['batting']
    team2_score = base_score + ((team2_rating - 70) * 2) + weather_adj['batting']
    
    # Add randomness
    team1_score += random.randint(-score_range//3, score_range//3)
    team2_score += random.randint(-score_range//3, score_range//3)
    
    # Ensure minimums
    min_scores = {'T20': 120, 'ODI': 200, 'Test': 250}
    team1_score = max(min_scores[match_format], team1_score)
    team2_score = max(min_scores[match_format], team2_score)
    
    print(f"🏏 Expected {team1} Score: {team1_score} ± 25")
    print(f"🏏 Expected {team2} Score: {team2_score} ± 25")
    print(f"🌤️  Weather Impact: {weather_adj['batting']:+d} runs, {weather_adj['bowling']}")
    
    # 3. KEY PREDICTIONS
    print(f"\n🎲 KEY MATCH EVENTS")
    print("=" * 20)
    
    if match_format == 'T20':
        events = [
            f"🎯 Powerplay Score: {45 + random.randint(-10, 15)} runs",
            f"🏃 Top Scorer: {40 + random.randint(0, 30)} runs",
            f"🎳 Death Overs: {50 + random.randint(-15, 25)} runs",
            f"⚡ Fastest Fifty: Over {8 + random.randint(-2, 4)}"
        ]
    elif match_format == 'ODI':
        events = [
            f"🎯 Powerplay Score: {55 + random.randint(-15, 20)} runs",
            f"💯 Century Chance: {random.choice(['High (70%)', 'Medium (45%)', 'Low (25%)'])}",
            f"🤝 Highest Partnership: {85 + random.randint(-25, 40)} runs",
            f"🎳 Most Wickets To: {random.choice(['Fast bowlers', 'Spinners', 'All-rounders'])}"
        ]
    else:  # Test
        events = [
            f"📅 Day 1 Score: {280 + random.randint(-50, 70)} runs",
            f"🏏 First Innings Lead: {45 + random.randint(-30, 60)} runs", 
            f"📊 Total Match Runs: {950 + random.randint(-200, 300)}",
            f"🎯 Result: {random.choice(['Win by innings', 'Win by 150+ runs', 'Close finish'])}"
        ]
    
    for event in events:
        print(f"   {event}")
    
    # 4. LIVE MATCH ANALYSIS (if current match data provided)
    if current_score is not None and current_over is not None:
        print(f"\n🔴 LIVE MATCH STATUS")
        print("=" * 22)
        
        wickets = current_wickets or 0
        current_rr = current_score / current_over if current_over > 0 else 0
        
        print(f"📺 Current: {current_score}/{wickets} ({current_over} overs)")
        print(f"📈 Run Rate: {current_rr:.2f}")
        
        # Projection
        total_overs = {'T20': 20, 'ODI': 50, 'Test': 90}[match_format]
        overs_left = total_overs - current_over
        
        if wickets <= 7:
            acceleration = 1.15
        else:
            acceleration = 0.95
            
        projected = current_score + (current_rr * overs_left * acceleration)
        print(f"🎯 Projected Final: {int(projected)}")
        
        # Win probability if chasing
        if target:
            needed = target - current_score
            req_rr = needed / overs_left if overs_left > 0 else 99
            
            if req_rr <= current_rr * 1.1:
                win_prob = 75
            elif req_rr <= current_rr * 1.4:
                win_prob = 45
            else:
                win_prob = 20
                
            win_prob = max(10, min(90, win_prob - (wickets * 5)))
            
            print(f"🎯 Target: {target} | Need: {needed} runs")
            print(f"📊 Required RR: {req_rr:.2f}")
            print(f"🏆 Win Probability: {win_prob}%")
    
    # Return structured results
    results = {
        'winner': winner,
        'team1_probability': team1_prob,
        'team2_probability': team2_prob,
        'confidence': confidence,
        'team1_predicted_score': team1_score,
        'team2_predicted_score': team2_score,
        'venue_advantage': venue_advantage,
        'weather_impact': weather_adj,
        'key_events': events
    }
    
    if current_score is not None:
        results.update({
            'current_status': f"{current_score}/{current_wickets or 0}",
            'projected_final': int(projected) if 'projected' in locals() else None,
            'win_probability_live': win_prob if target else None
        })
    
    print(f"\n✅ PREDICTION COMPLETE!")
    print("🏏" + "="*60 + "🏏")
    
    return results

def main():
    """Main function to run the cricket analysis"""
    # Configuration Settings
    MAX_MATCHES_TO_LOAD = 50
    MATCH_FORMAT_FILTER = None
    TEAM_FILTER = None
    
    # Download data if not already present
    if not os.path.exists("cricsheet_data"):
        cricsheet_url = "https://cricsheet.org/downloads/all_male_csv.zip"
        data_directory, available_files = download_cricsheet_data(cricsheet_url)
    else:
        available_files = glob("cricsheet_data/*.csv")
    
    # Load sample data for demonstration
    if len(available_files) > 0:
        sample_file = available_files[0]
        match_info, players, ball_df = parse_cricket_csv(sample_file)
        
        # Create features
        featured_data, encoders = create_features(ball_df, match_info)
        
        # Train models
        score_model, score_features, score_results = build_score_prediction_model(featured_data)
        
        # Create predictor with or without ML models
        if score_model is not None:
            predictor = CricketMatchPredictor(score_model=score_model)
        else:
            predictor = CricketMatchPredictor()
        
        # Interactive predictions
        print("� CRICKET PREDICTION SYSTEM READY!")
        print("=" * 40)
        
        # Ask if user wants to make predictions
        try:
            start_predictions = input("Would you like to start making predictions? (y/n): ").lower()
            if start_predictions == 'y':
                demo_predictions()
        except KeyboardInterrupt:
            pass
    else:
        print("❌ No cricket data found. Please check the data directory.")

def demo_predictions():
    """Interactive demo function for user input"""
    print("🏏 CRICKET PREDICTION SYSTEM")
    print("=" * 50)
    
    # Available teams
    teams = ['India', 'Australia', 'England', 'New Zealand', 'South Africa', 
             'Pakistan', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan',
             'Zimbabwe', 'Ireland', 'Netherlands']
    
    print("Available Teams:")
    for i, team in enumerate(teams, 1):
        print(f"{i:2d}. {team}")
    
    print("\n" + "="*50)
    
    while True:
        try:
            print("\n🎯 ENTER MATCH DETAILS:")
            print("-" * 30)
            
            # Get team selections
            team1_num = int(input("Select Team 1 (number): "))
            team2_num = int(input("Select Team 2 (number): "))
            
            if team1_num < 1 or team1_num > len(teams) or team2_num < 1 or team2_num > len(teams):
                print("❌ Invalid team number. Please select from 1 to", len(teams))
                continue
                
            team1 = teams[team1_num - 1]
            team2 = teams[team2_num - 1]
            
            if team1 == team2:
                print("❌ Please select different teams!")
                continue
            
                                     # Match format is fixed to Test
            match_format = 'Test'
            
            # Get venue
            print(f"\nVenue:")
            print("1. Neutral")
            print(f"2. {team1} Home")
            print(f"3. {team2} Home")
            venue_choice = int(input("Select venue (1-3): "))
            
            if venue_choice == 2:
                venue = team1
            elif venue_choice == 3:
                venue = team2
            else:
                venue = 'neutral'
            
            # Get weather
            print("\nWeather:")
            print("1. Clear")
            print("2. Overcast")
            print("3. Sunny")
            print("4. Windy")
            print("5. Humid")
            weather_choice = int(input("Select weather (1-5): "))
            
            weather_options = ['clear', 'overcast', 'sunny', 'windy', 'humid']
            if 1 <= weather_choice <= 5:
                weather = weather_options[weather_choice - 1]
            else:
                weather = 'clear'
            
            # Check if user wants live match tracking
            print("\nLive Match Tracking:")
            print("1. Pre-match prediction only")
            print("2. Include live match tracking")
            tracking_choice = int(input("Select option (1-2): "))
            
            current_score = None
            current_wickets = None
            current_over = None
            target = None
            
            if tracking_choice == 2:
                print("\n📺 LIVE MATCH DATA:")
                try:
                    current_score = int(input("Current score: "))
                    current_wickets = int(input("Current wickets: "))
                    current_over = float(input("Current over (e.g., 25.3): "))
                    
                    # Ask if chasing
                    chase_choice = input("Is team chasing? (y/n): ").lower()
                    if chase_choice == 'y':
                        target = int(input("Target score: "))
                except ValueError:
                    print("❌ Invalid input. Using pre-match prediction only.")
            
            # Make prediction
            print("\n" + "🏏"*25)
            print(f"PREDICTION: {team1} vs {team2}")
            print(f"Format: {match_format} | Venue: {venue} | Weather: {weather}")
            print("🏏"*25)
            
            result = predict_cricket_match(
                team1, team2, match_format, venue, weather,
                current_score, current_wickets, current_over, target
            )
            
            # Ask if user wants another prediction
            print("\n" + "="*50)
            another = input("Make another prediction? (y/n): ").lower()
            if another != 'y':
                break
                
        except ValueError:
            print("❌ Invalid input. Please enter numbers only.")
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using Cricket Prediction System!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")
    
    print("\n✅ Cricket prediction system demo complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"⚠️  Error: {e}")
        demo_predictions() 