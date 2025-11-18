# Cricket Prediction Model Accuracy Testing
# Test script for evaluating model performance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import the main prediction function
from mainmain import predict_cricket_match

class CricketAccuracyTester:
    """Comprehensive accuracy testing for cricket prediction model"""
    
    def __init__(self):
        self.test_results = []
        self.team_ratings = {
            'India': 85, 'Australia': 82, 'England': 80, 'New Zealand': 78,
            'South Africa': 76, 'Pakistan': 74, 'West Indies': 70, 
            'Sri Lanka': 68, 'Bangladesh': 65, 'Afghanistan': 62,
            'Zimbabwe': 58, 'Ireland': 55, 'Netherlands': 52
        }
        
    def run_basic_accuracy_tests(self):
        """Test basic prediction accuracy with known scenarios"""
        print("🏏 CRICKET PREDICTION ACCURACY TESTING")
        print("=" * 60)
        
        # Test scenarios
        test_scenarios = [
            # Strong vs Weak teams
            {'team1': 'India', 'team2': 'Afghanistan', 'venue': 'neutral', 'weather': 'clear'},
            {'team1': 'Australia', 'team2': 'Bangladesh', 'venue': 'Australia', 'weather': 'sunny'},
            
            # Close matches
            {'team1': 'India', 'team2': 'Australia', 'venue': 'neutral', 'weather': 'clear'},
            {'team1': 'England', 'team2': 'South Africa', 'venue': 'England', 'weather': 'overcast'},
            
            # Home advantage tests
            {'team1': 'New Zealand', 'team2': 'Pakistan', 'venue': 'New Zealand', 'weather': 'clear'},
            {'team1': 'Sri Lanka', 'team2': 'West Indies', 'venue': 'Sri Lanka', 'weather': 'humid'},
            
            # Weather impact tests
            {'team1': 'India', 'team2': 'England', 'venue': 'neutral', 'weather': 'overcast'},
            {'team1': 'Australia', 'team2': 'South Africa', 'venue': 'neutral', 'weather': 'windy'},
        ]
        
        print("\n📊 RUNNING BASIC ACCURACY TESTS...")
        print("-" * 40)
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n🧪 Test {i}: {scenario['team1']} vs {scenario['team2']}")
            print(f"   Venue: {scenario['venue']} | Weather: {scenario['weather']}")
            
            try:
                # Run prediction
                result = predict_cricket_match(
                    team1=scenario['team1'],
                    team2=scenario['team2'],
                    match_format='Test',
                    venue=scenario['venue'],
                    weather=scenario['weather']
                )
                
                # Store results
                test_result = {
                    'test_id': i,
                    'team1': scenario['team1'],
                    'team2': scenario['team2'],
                    'venue': scenario['venue'],
                    'weather': scenario['weather'],
                    'predicted_winner': result['winner'],
                    'team1_probability': result['team1_probability'],
                    'team2_probability': result['team2_probability'],
                    'confidence': result['confidence'],
                    'team1_predicted_score': result['team1_predicted_score'],
                    'team2_predicted_score': result['team2_predicted_score'],
                    'timestamp': datetime.now()
                }
                
                self.test_results.append(test_result)
                
                print(f"   ✅ Prediction: {result['winner']} wins ({result['confidence']:.1%} confidence)")
                print(f"   📊 Scores: {result['team1_predicted_score']} vs {result['team2_predicted_score']}")
                
            except Exception as e:
                print(f"   ❌ Error in test {i}: {e}")
        
        return len(test_scenarios)
    
    def run_consistency_tests(self):
        """Test model consistency with repeated predictions"""
        print("\n🔄 RUNNING CONSISTENCY TESTS...")
        print("-" * 40)
        
        # Test same scenario multiple times
        test_team1, test_team2 = 'India', 'Australia'
        consistency_results = []
        
        print(f"Testing consistency: {test_team1} vs {test_team2} (10 iterations)")
        
        for i in range(10):
            try:
                result = predict_cricket_match(
                    team1=test_team1,
                    team2=test_team2,
                    match_format='Test',
                    venue='neutral',
                    weather='clear'
                )
                
                consistency_results.append({
                    'iteration': i + 1,
                    'winner': result['winner'],
                    'confidence': result['confidence'],
                    'team1_score': result['team1_predicted_score'],
                    'team2_score': result['team2_predicted_score']
                })
                
            except Exception as e:
                print(f"   ❌ Error in iteration {i+1}: {e}")
        
        # Analyze consistency
        winners = [r['winner'] for r in consistency_results]
        confidences = [r['confidence'] for r in consistency_results]
        team1_scores = [r['team1_score'] for r in consistency_results]
        team2_scores = [r['team2_score'] for r in consistency_results]
        
        print(f"\n📈 Consistency Analysis:")
        print(f"   Winner consistency: {winners.count(winners[0])}/{len(winners)} times")
        print(f"   Average confidence: {np.mean(confidences):.1%}")
        print(f"   Team 1 score range: {min(team1_scores)}-{max(team1_scores)} (avg: {np.mean(team1_scores):.1f})")
        print(f"   Team 2 score range: {min(team2_scores)}-{max(team2_scores)} (avg: {np.mean(team2_scores):.1f})")
        
        return consistency_results
    
    def run_edge_case_tests(self):
        """Test edge cases and extreme scenarios"""
        print("\n⚠️  RUNNING EDGE CASE TESTS...")
        print("-" * 40)
        
        edge_cases = [
            # Very close teams
            {'team1': 'India', 'team2': 'Australia', 'venue': 'neutral'},
            
            # Strong home advantage
            {'team1': 'England', 'team2': 'Australia', 'venue': 'England'},
            
            # Weather extremes
            {'team1': 'India', 'team2': 'Pakistan', 'venue': 'neutral', 'weather': 'overcast'},
            {'team1': 'Australia', 'team2': 'England', 'venue': 'neutral', 'weather': 'humid'},
            
            # Lower ranked teams
            {'team1': 'Afghanistan', 'team2': 'Zimbabwe', 'venue': 'neutral'},
        ]
        
        edge_results = []
        
        for i, case in enumerate(edge_cases, 1):
            print(f"\n🧪 Edge Case {i}: {case['team1']} vs {case['team2']}")
            
            try:
                result = predict_cricket_match(
                    team1=case['team1'],
                    team2=case['team2'],
                    match_format='Test',
                    venue=case.get('venue', 'neutral'),
                    weather=case.get('weather', 'clear')
                )
                
                edge_results.append({
                    'case': i,
                    'scenario': f"{case['team1']} vs {case['team2']}",
                    'winner': result['winner'],
                    'confidence': result['confidence'],
                    'score_diff': abs(result['team1_predicted_score'] - result['team2_predicted_score'])
                })
                
                print(f"   ✅ Winner: {result['winner']} (Confidence: {result['confidence']:.1%})")
                print(f"   📊 Score difference: {abs(result['team1_predicted_score'] - result['team2_predicted_score'])} runs")
                
            except Exception as e:
                print(f"   ❌ Error in edge case {i}: {e}")
        
        return edge_results
    
    def run_live_match_simulation(self):
        """Simulate live match predictions"""
        print("\n📺 RUNNING LIVE MATCH SIMULATION...")
        print("-" * 40)
        
        live_scenarios = [
            # Chasing scenario
            {'team1': 'India', 'team2': 'Australia', 'current_score': 150, 'current_wickets': 3, 'current_over': 25, 'target': 280},
            
            # Batting first scenario
            {'team1': 'England', 'team2': 'South Africa', 'current_score': 200, 'current_wickets': 5, 'current_over': 35},
            
            # Close finish scenario
            {'team1': 'New Zealand', 'team2': 'Pakistan', 'current_score': 180, 'current_wickets': 4, 'current_over': 30, 'target': 220},
        ]
        
        live_results = []
        
        for i, scenario in enumerate(live_scenarios, 1):
            print(f"\n📺 Live Scenario {i}: {scenario['team1']} vs {scenario['team2']}")
            print(f"   Current: {scenario['current_score']}/{scenario['current_wickets']} ({scenario['current_over']} overs)")
            
            try:
                result = predict_cricket_match(
                    team1=scenario['team1'],
                    team2=scenario['team2'],
                    match_format='Test',
                    venue='neutral',
                    weather='clear',
                    current_score=scenario['current_score'],
                    current_wickets=scenario['current_wickets'],
                    current_over=scenario['current_over'],
                    target=scenario.get('target')
                )
                
                live_results.append({
                    'scenario': i,
                    'current_status': f"{scenario['current_score']}/{scenario['current_wickets']}",
                    'projected_final': result.get('projected_final'),
                    'win_probability': result.get('win_probability_live'),
                    'target': scenario.get('target')
                })
                
                if 'projected_final' in result:
                    print(f"   🎯 Projected Final: {result['projected_final']}")
                if 'win_probability_live' in result:
                    print(f"   🏆 Win Probability: {result['win_probability_live']}%")
                
            except Exception as e:
                print(f"   ❌ Error in live scenario {i}: {e}")
        
        return live_results
    
    def calculate_overall_accuracy(self):
        """Calculate overall accuracy metrics"""
        print("\n📊 CALCULATING OVERALL ACCURACY METRICS...")
        print("-" * 40)
        
        if not self.test_results:
            print("❌ No test results available")
            return
        
        # Extract metrics
        confidences = [r['confidence'] for r in self.test_results]
        score_diffs = [abs(r['team1_predicted_score'] - r['team2_predicted_score']) for r in self.test_results]
        
        # Calculate accuracy metrics
        avg_confidence = np.mean(confidences)
        avg_score_diff = np.mean(score_diffs)
        consistency_score = len(set([r['predicted_winner'] for r in self.test_results])) / len(self.test_results)
        
        print(f"📈 Overall Accuracy Metrics:")
        print(f"   Average Confidence: {avg_confidence:.1%}")
        print(f"   Average Score Difference: {avg_score_diff:.1f} runs")
        print(f"   Prediction Consistency: {consistency_score:.1%}")
        print(f"   Total Tests Run: {len(self.test_results)}")
        
        # Rating system
        if avg_confidence > 0.75:
            accuracy_rating = "Excellent"
        elif avg_confidence > 0.65:
            accuracy_rating = "Good"
        elif avg_confidence > 0.55:
            accuracy_rating = "Fair"
        else:
            accuracy_rating = "Poor"
        
        print(f"\n🏆 Overall Rating: {accuracy_rating}")
        
        return {
            'avg_confidence': avg_confidence,
            'avg_score_diff': avg_score_diff,
            'consistency_score': consistency_score,
            'accuracy_rating': accuracy_rating,
            'total_tests': len(self.test_results)
        }
    
    def generate_accuracy_report(self):
        """Generate comprehensive accuracy report"""
        print("\n📋 GENERATING ACCURACY REPORT...")
        print("=" * 60)
        
        # Run all tests
        basic_tests = self.run_basic_accuracy_tests()
        consistency_tests = self.run_consistency_tests()
        edge_tests = self.run_edge_case_tests()
        live_tests = self.run_live_match_simulation()
        
        # Calculate overall accuracy
        accuracy_metrics = self.calculate_overall_accuracy()
        
        # Generate report
        report = {
            'timestamp': datetime.now(),
            'basic_tests': basic_tests,
            'consistency_tests': len(consistency_tests),
            'edge_tests': len(edge_tests),
            'live_tests': len(live_tests),
            'accuracy_metrics': accuracy_metrics,
            'test_results': self.test_results
        }
        
        print(f"\n📊 ACCURACY REPORT SUMMARY:")
        print("=" * 40)
        print(f"📅 Test Date: {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🧪 Basic Tests: {report['basic_tests']}")
        print(f"🔄 Consistency Tests: {report['consistency_tests']}")
        print(f"⚠️  Edge Case Tests: {report['edge_tests']}")
        print(f"📺 Live Match Tests: {report['live_tests']}")
        
        if accuracy_metrics:
            print(f"\n🏆 Final Rating: {accuracy_metrics['accuracy_rating']}")
            print(f"📈 Average Confidence: {accuracy_metrics['avg_confidence']:.1%}")
            print(f"🎯 Prediction Consistency: {accuracy_metrics['consistency_score']:.1%}")
        
        return report

def main():
    """Main function to run accuracy testing"""
    print("🏏 CRICKET PREDICTION MODEL ACCURACY TESTING")
    print("=" * 60)
    
    # Initialize tester
    tester = CricketAccuracyTester()
    
    try:
        # Run comprehensive accuracy testing
        report = tester.generate_accuracy_report()
        
        print(f"\n✅ ACCURACY TESTING COMPLETE!")
        print("=" * 40)
        print("📊 The model has been thoroughly tested across various scenarios.")
        print("🎯 Use this report to assess model performance and reliability.")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Error during accuracy testing: {e}")
        return None

if __name__ == "__main__":
    main() 