from flask import Flask, render_template, request, jsonify, send_file
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from mainmain import predict_cricket_match
from data_updater import CricketDataUpdater
import os

app = Flask(__name__)

# Initialize data updater
data_updater = CricketDataUpdater()

# Check for data updates on app startup (non-blocking)
def check_for_updates():
    """Check for data updates in background"""
    try:
        if data_updater.should_update():
            print("🏏 Checking for cricket data updates...")
            # Run update in background thread to avoid blocking app startup
            import threading
            update_thread = threading.Thread(target=lambda: data_updater.update_data_and_models())
            update_thread.daemon = True
            update_thread.start()
            print("🚀 Data update started in background...")
    except Exception as e:
        print(f"⚠️  Data update check failed: {e}")

# Check for updates when app starts
check_for_updates()

# Available teams
TEAMS = [
    'India', 'Australia', 'England', 'New Zealand', 'South Africa', 
    'Pakistan', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan',
    'Zimbabwe', 'Ireland', 'Netherlands'
]

FORMATS = ['T20', 'ODI', 'Test']
WEATHER_OPTIONS = ['clear', 'overcast', 'sunny', 'windy', 'humid']

@app.route('/')
def index():
    """Home page with prediction form"""
    return render_template('index.html', teams=TEAMS, formats=FORMATS, weather_options=WEATHER_OPTIONS)

@app.route('/data-manager')
def data_manager():
    """Data management interface"""
    return render_template('data_manager.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.get_json()
        
        team1 = data.get('team1')
        team2 = data.get('team2')
        match_format = data.get('format', 'ODI')
        venue = data.get('venue', 'neutral')
        weather = data.get('weather', 'clear')
        
        # Live match data (optional)
        current_score = data.get('current_score')
        current_wickets = data.get('current_wickets')
        current_over = data.get('current_over')
        target = data.get('target')
        
        # Convert string inputs to appropriate types
        if current_score:
            current_score = int(current_score)
        if current_wickets:
            current_wickets = int(current_wickets)
        if current_over:
            current_over = float(current_over)
        if target:
            target = int(target)
        
        # Validate teams
        if team1 not in TEAMS or team2 not in TEAMS:
            return jsonify({'error': 'Invalid team selection'}), 400
        
        if team1 == team2:
            return jsonify({'error': 'Please select different teams'}), 400
        
        # Make prediction (suppress console output for web interface)
        import sys
        from io import StringIO
        
        # Temporarily redirect stdout to suppress prints
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            result = predict_cricket_match(
                team1, team2, match_format, venue, weather,
                current_score, current_wickets, current_over, target
            )
        finally:
            # Restore stdout
            sys.stdout = old_stdout
        
        # Create visualization
        chart_url = create_prediction_chart(result, team1, team2)
        result['chart_url'] = chart_url
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page with information about the prediction system"""
    return render_template('about.html')

@app.route('/stats')
def stats():
    """Statistics page with model performance and cricket analytics"""
    return render_template('stats.html')

@app.route('/api/teams')
def get_teams():
    """API endpoint to get available teams"""
    return jsonify(TEAMS)

def create_prediction_chart(result, team1, team2):
    """Create a visualization chart for the prediction results"""
    try:
        # Set up the plot
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Cricket Match Prediction: {team1} vs {team2}', fontsize=16, fontweight='bold')
        
        # 1. Win Probability Chart
        teams = [team1, team2]
        probabilities = [result['team1_probability'] * 100, result['team2_probability'] * 100]
        colors = ['#2E86AB', '#A23B72']
        
        bars1 = ax1.bar(teams, probabilities, color=colors, alpha=0.8)
        ax1.set_title('Win Probability (%)', fontweight='bold')
        ax1.set_ylabel('Probability (%)')
        ax1.set_ylim(0, 100)
        
        # Add percentage labels on bars
        for bar, prob in zip(bars1, probabilities):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Score Prediction Chart
        scores = [result['team1_predicted_score'], result['team2_predicted_score']]
        bars2 = ax2.bar(teams, scores, color=colors, alpha=0.8)
        ax2.set_title('Predicted Scores', fontweight='bold')
        ax2.set_ylabel('Runs')
        
        # Add score labels on bars
        for bar, score in zip(bars2, scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{score}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Confidence Meter
        confidence = result['confidence'] * 100
        ax3.pie([confidence, 100-confidence], labels=['Confidence', ''], 
                colors=['#F18F01', '#E8E8E8'], startangle=90,
                wedgeprops=dict(width=0.3))
        ax3.set_title(f'Prediction Confidence: {confidence:.1f}%', fontweight='bold')
        
        # 4. Key Statistics
        ax4.axis('off')
        stats_text = f"""
        Match Details:
        
        🏆 Predicted Winner: {result['winner']}
        🎯 Confidence Level: {confidence:.1f}%
        🏟️ Venue Advantage: {result['venue_advantage']}
        🌤️ Weather Impact: {result['weather_impact']['bowling']}
        
        Score Predictions:
        📊 {team1}: {result['team1_predicted_score']} runs
        📊 {team2}: {result['team2_predicted_score']} runs
        
        💡 Betting Advice:
        {result['betting_advice']}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        # Save to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

@app.route('/api/data-status')
def data_status():
    """Get current data status and last update info"""
    try:
        config = data_updater.load_config()
        stats = data_updater.get_current_data_stats()
        
        return jsonify({
            'last_update': config.get('last_update'),
            'should_update': data_updater.should_update(),
            'total_matches': stats['total_matches'],
            'date_range': stats['date_range'],
            'update_frequency_hours': config.get('update_frequency_hours', 24)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-data', methods=['POST'])
def trigger_update():
    """Manually trigger data update"""
    try:
        force = request.json.get('force', False) if request.is_json else False
        
        # Run update in background thread
        import threading
        
        def run_update():
            success = data_updater.update_data_and_models(force=force)
            print(f"Manual update {'completed' if success else 'failed'}")
        
        update_thread = threading.Thread(target=run_update)
        update_thread.daemon = True
        update_thread.start()
        
        return jsonify({
            'message': 'Data update started in background',
            'status': 'started'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
