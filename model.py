import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

class FantasyPredictor:
    def __init__(self):
        self.model = None
        self.player_encoder = LabelEncoder()
        self.pos_encoder = LabelEncoder()
        self.features = ['Player_ID', 'Age', 'Pos_ID']
        self.is_trained = False

    def train_model(self, filename='complete.csv'):
        """
        Loads data from CSV, cleans it, generates weights, and trains the Random Forest.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Error: {filename} not found.")

        # 1. Load Data
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()

        # 2. Identify Target Column
        target_col = None
        if 'FantPT' in df.columns:
            target_col = 'FantPT'
        elif 'FantPt' in df.columns:
            target_col = 'FantPt'
        else:
            raise ValueError("Error: Could not find 'FantPt' or 'FantPT' column.")

        # 3. Data Cleaning
        # Clean Player Names
        df['Player'] = df['Player'].astype(str).apply(lambda x: x.split('*')[0].split('+')[0].strip())
        
        # Clean Position and Age
        df['FantPos'] = df['FantPos'].astype(str)
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(25)

        # Drop invalid rows
        df = df.dropna(subset=[target_col])
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])

        #print(f"Data ready. Training on {len(df)} rows.")

        # 4. Feature Engineering
        # Fit and transform encoders
        df['Player_ID'] = self.player_encoder.fit_transform(df['Player'])
        df['Pos_ID'] = self.pos_encoder.fit_transform(df['FantPos'])

        # 5. Weighting Logic
        current_season = 2025
        df['Year_Gap'] = current_season - df['Year'] 
        # Exponential decay: recent years matter significantly more
        df['Weight'] = 1 / (df['Year_Gap'].clip(lower=0.5) ** 2)
        df['Weight'] = df['Weight'] / df['Weight'].mean()

        # 6. Train Model
        X = df[self.features]
        y = df[target_col]
        weights = df['Weight']

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y, sample_weight=weights)
        
        self.is_trained = True
        print("Model trained successfully!")

    def predict_player(self, player_name, age_next_season, position):
        """
        Predicts fantasy points for a specific player/age/pos combo.
        Returns None if player or position is unknown.
        """
        if not self.is_trained:
            print("Error: Model is not trained. Call train_model() first.")
            return None

        clean_name = player_name.split('*')[0].split('+')[0].strip()
        
        try:
            # Transform inputs using the encoders saved during training
            player_id = self.player_encoder.transform([clean_name])[0]
            pos_id = self.pos_encoder.transform([position])[0]
            
            # Create input DataFrame to match training feature names
            input_df = pd.DataFrame(
                [[player_id, age_next_season, pos_id]], 
                columns=self.features
            )
            
            prediction = self.model.predict(input_df)[0]
            return float(prediction)
            
        except ValueError:
            # This handles cases where the Player Name or Position wasn't in the training CSV
            return None

# ---------------------------------------------------------
# EXECUTION BLOCK
# This only runs if you run this file directly.
# It does NOT run if you import this file.
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialize
    predictor = FantasyPredictor()
    
    # 2. Train
    try:
        predictor.train_model('complete.csv')
        
        # 3. Test
        print(f"\n{'PLAYER':<20} | {'AGE':<4} | {'POS':<4} | {'PREDICTION':<10}")
        print("-" * 55)

        test_cases = [
            ("Patrick Mahomes", 29, "QB"),
            ("Derrick Henry", 31, "RB"),
            ("Justin Jefferson", 25, "WR"),
            ("CeeDee Lamb", 25, "WR"),
            ("Rookie Player", 21, "WR") # Should return None/Not Found
        ]

        for p, age, pos in test_cases:
            pred = predictor.predict_player(p, age, pos)
            if pred is not None:
                print(f"{p:<20} | {age:<4} | {pos:<4} | {pred:.2f}")
            else:
                print(f"{p:<20} | Not found in training data")
                
    except Exception as e:
        print(e)