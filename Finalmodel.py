import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import chardet
import re
import warnings
warnings.filterwarnings('ignore')

# Function to clean funding amounts
def clean_funding_amount(amount):
    if pd.isna(amount):
        return np.nan
    if isinstance(amount, (int, float)):
        return amount

    amount_str = str(amount).strip()
    if re.match(r'^\d{1,2},\d{2},\d{3}$', amount_str):
        parts = amount_str.split(',')
        if len(parts) == 3:
            lakhs = int(parts[0])
            thousands = int(parts[1])
            remainder = int(parts[2])
            return (lakhs * 100000) + (thousands * 1000) + remainder

    clean_str = amount_str.replace(',', '')
    try:
        return float(clean_str)
    except (ValueError, TypeError):
        return np.nan

# Identify outliers
def identify_outliers(data, column, threshold=3):
    mean = data[column].mean()
    std = data[column].std()
    z_scores = (data[column] - mean) / std
    return data[abs(z_scores) > threshold].index

# Load dataset
def load_data(filename="D1.csv"):
    print(f"Loading dataset from {filename}...")
    try:
        with open(filename, 'rb') as f:
            result = chardet.detect(f.read())
            encoding_used = result['encoding']
        
        df = pd.read_csv(filename, encoding=encoding_used)
        print(f"Successfully loaded dataset with {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check the file path and try again.")
        exit(1)

# Process data
def process_data(df):
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    j
    # Map common column names
    column_mapping = {
        "funding_total_usd": "funding_total_usd",
        "total_funding": "funding_total_usd",
        "funding_amount": "funding_total_usd",
        "founded_at": "founded_at",
        "market": "market",
        "country_code": "country_code",
        "funding_rounds": "funding_rounds",
        "category_list": "category_list",
        "state_code": "state_code",
        "city": "city",
        "first_funding_at": "first_funding_at",
        "last_funding_at": "last_funding_at"
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    # Convert funding to numeric
    df["funding_total_usd"] = df["funding_total_usd"].apply(clean_funding_amount)
    df["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors='coerce')
    
    # Time-based features
    df["founded_at"] = pd.to_datetime(df["founded_at"], errors='coerce')
    df["first_funding_at"] = pd.to_datetime(df["first_funding_at"], errors='coerce')
    df["last_funding_at"] = pd.to_datetime(df["last_funding_at"], errors='coerce')
    
    df["years_since_founded"] = (pd.to_datetime("today") - df["founded_at"]).dt.days / 365.25
    df["time_to_first_funding"] = (df["first_funding_at"] - df["founded_at"]).dt.days / 365.25
    df["funding_duration"] = (df["last_funding_at"] - df["first_funding_at"]).dt.days / 365.25
    
    # Advanced features
    df["funding_velocity"] = df["funding_rounds"] / df["funding_duration"].replace(0, 0.1)
    df["avg_funding_per_round"] = df["funding_total_usd"] / df["funding_rounds"].replace(0, 1)
    df["funding_per_year"] = df["funding_total_usd"] / df["years_since_founded"].replace(0, 0.1)
    df["quick_funding"] = (df["time_to_first_funding"] < 1).astype(int)
    
    # Location features
    if "state_code" in df.columns:
        df["has_state"] = df["state_code"].notna() & (df["state_code"] != "")
    else:
        df["has_state"] = False
    
    # Country grouping
    country_counts = df["country_code"].value_counts()
    common_countries = country_counts[country_counts >= 10].index
    df["country_group"] = df["country_code"].apply(lambda x: x if x in common_countries else "OTHER")
    
    # Market features
    if "market" in df.columns:
        df["market_main"] = df["market"].fillna("").astype(str).apply(lambda x: x.split()[0] if len(x.split()) > 0 else "Unknown")
        tech_markets = ['Software', 'Mobile', 'Internet', 'Enterprise', 'Web', 'Cloud', 'SaaS', 'AI', 'Machine']
        health_markets = ['Health', 'Medical', 'Healthcare', 'Biotech', 'Pharma']
        finance_markets = ['Finance', 'FinTech', 'Insurance', 'Banking', 'Investment']
        
        def categorize_market(market):
            if market in tech_markets:
                return 'Tech'
            elif market in health_markets:
                return 'Health'
            elif market in finance_markets:
                return 'Finance'
            else:
                return 'Other'
        
        df['market_sector'] = df['market_main'].apply(categorize_market)
    
    # Handle missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["funding_rounds"] = df["funding_rounds"].fillna(0)
    df["years_since_founded"] = df["years_since_founded"].fillna(df["years_since_founded"].median())
    df["time_to_first_funding"] = df["time_to_first_funding"].fillna(0)
    df["funding_duration"] = df["funding_duration"].fillna(0)
    df["funding_velocity"] = df["funding_velocity"].fillna(0)
    df["avg_funding_per_round"] = df["avg_funding_per_round"].fillna(0)
    df["funding_per_year"] = df["funding_per_year"].fillna(0)
    
    # Filter valid funding data and remove outliers
    df_filtered = df[df["funding_total_usd"] > 0].copy()
    outlier_idx = identify_outliers(df_filtered, "funding_total_usd", threshold=4)
    df_cleaned = df_filtered.drop(outlier_idx)
    
    print(f"Data processing complete: {len(df_cleaned)} clean records after filtering.")
    return df_cleaned, country_counts

# Generate synthetic profitability data
def generate_profitability_data(df):
    df_copy = df.copy()
    
    funding_factor = np.log1p(df_copy['funding_total_usd']) / 20
    rounds_factor = df_copy['funding_rounds'] / 5
    quick_funding_factor = df_copy['quick_funding'] * 0.5
    
    sector_bonus = pd.Series(0, index=df_copy.index)
    if 'market_sector' in df_copy.columns:
        sector_bonus.loc[df_copy['market_sector'] == 'Tech'] = 0.2
        sector_bonus.loc[df_copy['market_sector'] == 'Health'] = -0.1
        sector_bonus.loc[df_copy['market_sector'] == 'Finance'] = 0.3
    
    profit_probability = 0.3 + (funding_factor + rounds_factor + quick_funding_factor + sector_bonus) / 10
    profit_probability = profit_probability.clip(0, 0.9)
    
    df_copy['is_profitable'] = np.random.binomial(1, profit_probability)
    
    revenue_multiplier = np.where(df_copy['is_profitable'] == 1,
                                np.random.uniform(1.5, 4.0, size=len(df_copy)),
                                np.random.uniform(0.2, 1.2, size=len(df_copy)))
    
    df_copy['estimated_revenue'] = df_copy['funding_total_usd'] * revenue_multiplier
    
    df_copy['profit_margin'] = np.where(df_copy['is_profitable'] == 1,
                                      np.random.uniform(0.05, 0.3, size=len(df_copy)),
                                      np.random.uniform(-0.5, 0.01, size=len(df_copy)))
    
    df_copy['profit_amount'] = df_copy['estimated_revenue'] * df_copy['profit_margin']
    
    return df_copy

# Train funding prediction model
def train_funding_model(df_cleaned):
    features = [
        "country_group", "market_sector", "funding_rounds", "years_since_founded",
        "time_to_first_funding", "funding_duration", "funding_velocity",
        "avg_funding_per_round", "funding_per_year", "quick_funding", "has_state"
    ]
    
    # Ensure all features exist
    for feature in features:
        if feature not in df_cleaned.columns:
            if feature in ["country_group", "market_sector"]:
                df_cleaned[feature] = "Unknown"
            else:
                df_cleaned[feature] = 0
    
    X = df_cleaned[features].copy()
    y = np.log1p(df_cleaned["funding_total_usd"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_features = ["country_group", "market_sector"]
    encoders = {}
    for col in categorical_features:
        if col in X.columns:
            encoders[col] = LabelEncoder()
            X_train[col] = encoders[col].fit_transform(X_train[col].astype(str))
            X_test[col] = encoders[col].transform(X_test[col].astype(str))
    
    numeric_features = [col for col in X.columns if col not in categorical_features]
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = xgb_model.predict(X_test)
    y_pred_exp = np.expm1(y_pred)
    y_test_exp = np.expm1(y_test)
    
    r2 = r2_score(y_test_exp, y_pred_exp)
    mae = mean_absolute_error(y_test_exp, y_pred_exp)
    median_ae = median_absolute_error(y_test_exp, y_pred_exp)
    
    print(f"Funding Model Performance - R² Score: {r2:.3f}")
    print(f"MAE: ${mae:,.2f}, Median AE: ${median_ae:,.2f}")
    
    return xgb_model, encoders, scaler, features, categorical_features, numeric_features

# Train profitability model
def train_profitability_model(df_with_outcomes, categorical_features):
    profitability_features = [
        "country_group", "market_sector", "funding_rounds",
        "years_since_founded", "time_to_first_funding",
        "funding_duration", "funding_velocity",
        "funding_total_usd", "avg_funding_per_round", "funding_per_year",
        "quick_funding", "has_state"
    ]
    
    # Ensure all features exist
    for feature in profitability_features:
        if feature not in df_with_outcomes.columns:
            if feature in categorical_features:
                df_with_outcomes[feature] = "Unknown"
            else:
                df_with_outcomes[feature] = 0
    
    X_profit = df_with_outcomes[profitability_features].copy()
    y_profit = df_with_outcomes["is_profitable"]
    
    profit_encoders = {}
    for col in categorical_features:
        if col in X_profit.columns:
            profit_encoders[col] = LabelEncoder()
            X_profit[col] = profit_encoders[col].fit_transform(X_profit[col].astype(str))
    
    X_profit_train, X_profit_test, y_profit_train, y_profit_test = train_test_split(
        X_profit, y_profit, test_size=0.2, random_state=42
    )
    
    profit_scaler = MinMaxScaler()
    numeric_profit_features = [col for col in X_profit.columns if col not in categorical_features]
    X_profit_train[numeric_profit_features] = profit_scaler.fit_transform(X_profit_train[numeric_profit_features])
    X_profit_test[numeric_profit_features] = profit_scaler.transform(X_profit_test[numeric_profit_features])
    
    profit_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    profit_model.fit(X_profit_train, y_profit_train)
    
    # Evaluate model
    y_profit_pred = profit_model.predict(X_profit_test)
    accuracy = accuracy_score(y_profit_test, y_profit_pred)
    
    print(f"Profitability Model - Accuracy: {accuracy:.3f}")
    
    return profit_model, profit_encoders, profit_scaler, profitability_features

# Predict funding for a single startup
def predict_funding(
    xgb_model, encoders, scaler, features, categorical_features, numeric_features,
    country_group, market_sector, funding_rounds, years_since_founded,
    time_to_first_funding, funding_duration, avg_funding_per_round=None
):
    if funding_velocity is None:
        funding_velocity = funding_rounds / max(funding_duration, 0.1)
    if quick_funding is None:
        quick_funding = 1 if time_to_first_funding < 1 else 0
    if avg_funding_per_round is None:
        avg_funding_per_round = 1000000  # Default $1M per round
    if funding_per_year is None:
        funding_per_year = (avg_funding_per_round * funding_rounds) / max(years_since_founded, 0.1)
    
    test_data = pd.DataFrame({
        "country_group": [country_group],
        "market_sector": [market_sector],
        "funding_rounds": [funding_rounds],
        "years_since_founded": [years_since_founded],
        "time_to_first_funding": [time_to_first_funding],
        "funding_duration": [funding_duration],
        "funding_velocity": [funding_velocity],
        "avg_funding_per_round": [avg_funding_per_round],
        "funding_per_year": [funding_per_year],
        "quick_funding": [quick_funding],
        "has_state": [has_state]
    })
    
    # Process test data
    for col, encoder in encoders.items():
        if col in test_data.columns:
            test_data[col] = test_data[col].astype(str)
            # Handle unseen categories
            for i, val in enumerate(test_data[col]):
                if val not in encoder.classes_:
                    test_data.loc[i, col] = encoder.classes_[0]
            test_data[col] = encoder.transform(test_data[col])
    
    if numeric_features:
        test_data[numeric_features] = scaler.transform(test_data[numeric_features])
    
    # Make prediction
    log_pred = xgb_model.predict(test_data[features])
    return np.expm1(log_pred)[0]

# Predict single startup profitability
def predict_profitability(
    profit_model, profit_encoders, profit_scaler, profitability_features, categorical_features,
    country_group, market_sector, funding_rounds, years_since_founded,
    time_to_first_funding, funding_duration, funding_total_usd,
    funding_velocity=None, avg_funding_per_round=None, funding_per_year=None,
    quick_funding=None, has_state=True
):
    # Calculate derived metrics if not provided
    if funding_velocity is None:
        funding_velocity = funding_rounds / max(funding_duration, 0.1)
    if quick_funding is None:
        quick_funding = 1 if time_to_first_funding < 1 else 0
    if avg_funding_per_round is None and funding_rounds > 0:
        avg_funding_per_round = funding_total_usd / funding_rounds
    else:
        avg_funding_per_round = funding_total_usd
    if funding_per_year is None:
        funding_per_year = funding_total_usd / max(years_since_founded, 0.1)
    
    # Create DataFrame for prediction
    test_data = pd.DataFrame({
        "country_group": [country_group],
        "market_sector": [market_sector],
        "funding_rounds": [funding_rounds],
        "years_since_founded": [years_since_founded],
        "time_to_first_funding": [time_to_first_funding],
        "funding_duration": [funding_duration],
        "funding_velocity": [funding_velocity],
        "funding_total_usd": [funding_total_usd],
        "avg_funding_per_round": [avg_funding_per_round],
        "funding_per_year": [funding_per_year],
        "quick_funding": [quick_funding],
        "has_state": [has_state]
    })
    
    # Process test data
    for col, encoder in profit_encoders.items():
        if col in test_data.columns:
            test_data[col] = test_data[col].astype(str)
            for i, val in enumerate(test_data[col]):
                if val not in encoder.classes_:
                    test_data.loc[i, col] = encoder.classes_[0]
            test_data[col] = encoder.transform(test_data[col])
    
    numeric_profit_features = [col for col in profitability_features if col not in categorical_features]
    if numeric_profit_features:
        test_data[numeric_profit_features] = profit_scaler.transform(test_data[numeric_profit_features])
    
    # Make prediction
    profit_score = profit_model.predict_proba(test_data[profitability_features])[:, 1][0]
    is_profitable = profit_model.predict(test_data[profitability_features])[0]
    
    # Calculate estimated financials
    if is_profitable == 1:
        profit_margin = 0.12 + (profit_score - 0.5) * 0.3
        estimated_revenue = funding_total_usd * (1.5 + profit_score)
    else:
        profit_margin = -0.2 - (0.5 - profit_score) * 0.6
        estimated_revenue = funding_total_usd * (0.7 + profit_score * 0.5)
    
    profit_amount = estimated_revenue * profit_margin
    
    return profit_score, is_profitable, profit_amount, profit_margin, estimated_revenue

# Interactive prediction function
def get_prediction():
    print("\n===== Startup Funding & Profitability Predictor =====")
    
    country_group = input("Country (e.g., USA, IND, GBR): ").strip().upper()
    
    market_options = ["Tech", "Health", "Finance", "Other"]
    print(f"Market sector options: {', '.join(market_options)}")
    market_sector = input("Market sector: ").strip()
    if market_sector not in market_options:
        print(f"Warning: Using '{market_sector}' as market sector")
    
    funding_rounds = int(input("Number of funding rounds: "))
    years_since_founded = float(input("Years since founding: "))
    time_to_first_funding = float(input("Years between founding and first funding: "))
    funding_duration = float(input("Years between first and last funding: "))
    
    avg_input = input("Average funding per round (USD, e.g., 1000000) [Enter for default]: ")
    avg_funding_per_round = float(avg_input) if avg_input.strip() else None
    
    has_state = True  # Default
    funding_velocity = funding_rounds / max(funding_duration, 0.1)
    quick_funding = 1 if time_to_first_funding < 1 else 0
    funding_per_year = None  # Will be calculated based on predicted funding
    
    # Predict funding
    predicted_funding = predict_single_startup(
        country_group, market_sector, funding_rounds,
        years_since_founded, time_to_first_funding,
        funding_duration, funding_velocity,
        avg_funding_per_round, funding_per_year, quick_funding, has_state
    )
    
    # Predict profitability
    profit_score, is_profitable, profit_amount, profit_margin, estimated_revenue = predict_single_startup_profitability(
        country_group, market_sector, funding_rounds,
        years_since_founded, time_to_first_funding,
        funding_duration, predicted_funding,
        funding_velocity, avg_funding_per_round,
        funding_per_year, quick_funding, has_state
    )
    
    # Display results
    print("\n===== Prediction Results =====")
    print(f"Predicted Total Funding: ${predicted_funding:,.2f}")
    print(f"Profitability Score (0-1): {profit_score:.2f}")
    print(f"Profitable: {'Yes' if is_profitable == 1 else 'No'}")
    print(f"Estimated Revenue: ${estimated_revenue:,.2f}")
    print(f"Predicted Profit Margin: {profit_margin:.1%}")
    print(f"Estimated Annual Profit/Loss: ${profit_amount:,.2f}")
    
    return predicted_funding, profit_score, is_profitable

# Predict single startup funding
def predict_single_startup(country_group, market_sector, funding_rounds,
                         years_since_founded, time_to_first_funding,
                         funding_duration, funding_velocity=None,
                         avg_funding_per_round=None, funding_per_year=None,
                         quick_funding=None, has_state=True):
    if funding_velocity is None:
        funding_velocity = funding_rounds / max(funding_duration, 0.1)
    if quick_funding is None:
        quick_funding = 1 if time_to_first_funding < 1 else 0
    if avg_funding_per_round is None:
        avg_funding_per_round = 1000000  # Default $1M per round
    if funding_per_year is None:
        funding_per_year = (avg_funding_per_round * funding_rounds) / max(years_since_founded, 0.1)

    test_data = pd.DataFrame({
        "country_group": [country_group],
        "market_sector": [market_sector],
        "funding_rounds": [funding_rounds],
        "years_since_founded": [years_since_founded],
        "time_to_first_funding": [time_to_first_funding],
        "funding_duration": [funding_duration],
        "funding_velocity": [funding_velocity],
        "avg_funding_per_round": [avg_funding_per_round],
        "funding_per_year": [funding_per_year],
        "quick_funding": [quick_funding],
        "has_state": [has_state]
    })

    for col, encoder in encoders.items():
        if col in test_data.columns:
            test_data[col] = test_data[col].astype(str)
            for i, val in enumerate(test_data[col]):
                if val not in encoder.classes_:
                    test_data.loc[i, col] = encoder.classes_[0]
            test_data[col] = encoder.transform(test_data[col])

    test_data[numeric_features] = scaler.transform(test_data[numeric_features])
    
    prediction = xgb_model.predict(test_data[features])
    return np.expm1(prediction)[0]

# Predict single startup profitability
def predict_single_startup_profitability(
    country_group, market_sector, funding_rounds,
    years_since_founded, time_to_first_funding,
    funding_duration, funding_total_usd,
    funding_velocity=None, avg_funding_per_round=None,
    funding_per_year=None, quick_funding=None, has_state=True
):
    # Calculate derived metrics if not provided
    if funding_velocity is None:
        funding_velocity = funding_rounds / max(funding_duration, 0.1)
    if quick_funding is None:
        quick_funding = 1 if time_to_first_funding < 1 else 0
    if avg_funding_per_round is None and funding_rounds > 0:
        avg_funding_per_round = funding_total_usd / funding_rounds
    else:
        avg_funding_per_round = funding_total_usd
    if funding_per_year is None:
        funding_per_year = funding_total_usd / max(years_since_founded, 0.1)

    # Create DataFrame for prediction
    test_data = pd.DataFrame({
        "country_group": [country_group],
        "market_sector": [market_sector],
        "funding_rounds": [funding_rounds],
        "years_since_founded": [years_since_founded],
        "time_to_first_funding": [time_to_first_funding],
        "funding_duration": [funding_duration],
        "funding_velocity": [funding_velocity],
        "funding_total_usd": [funding_total_usd],
        "avg_funding_per_round": [avg_funding_per_round],
        "funding_per_year": [funding_per_year],
        "quick_funding": [quick_funding],
        "has_state": [has_state]
    })

    # Process test data
    for col, encoder in profit_encoders.items():
        if col in test_data.columns:
            test_data[col] = test_data[col].astype(str)
            for i, val in enumerate(test_data[col]):
                if val not in encoder.classes_:
                    test_data.loc[i, col] = encoder.classes_[0]
            test_data[col] = encoder.transform(test_data[col])

    numeric_profit_features = [col for col in profitability_features if col not in categorical_features]
    test_data[numeric_profit_features] = profit_scaler.transform(test_data[numeric_profit_features])

    # Make prediction
    profit_score = profit_model.predict_proba(test_data[profitability_features])[:, 1][0]
    is_profitable = profit_model.predict(test_data[profitability_features])[0]

    # Calculate estimated financials
    if is_profitable == 1:
        profit_margin = 0.12 + (profit_score - 0.5) * 0.3
        estimated_revenue = funding_total_usd * (1.5 + profit_score)
    else:
        profit_margin = -0.2 - (0.5 - profit_score) * 0.6
        estimated_revenue = funding_total_usd * (0.7 + profit_score * 0.5)

    profit_amount = estimated_revenue * profit_margin

    return profit_score, is_profitable, profit_amount, profit_margin, estimated_revenue

# Main execution
if __name__ == "__main__":
    print("Startup Funding & Profitability Predictor")
    
    # Load the D1 dataset
    df = load_data("D1.csv")
    
    # Process data
    df_cleaned, country_counts = process_data(df)
    
    # Generate synthetic profitability data (since real profitability may not be available)
    df_with_outcomes = generate_profitability_data(df_cleaned)
    
    # Train models
    print("\nTraining funding prediction model...")
    xgb_model, encoders, scaler, features, categorical_features, numeric_features = train_funding_model(df_cleaned)
    
    print("\nTraining profitability prediction model...")
    profit_model, profit_encoders, profit_scaler, profitability_features = train_profitability_model(df_with_outcomes, categorical_features)
    
    # Interactive prediction
    print("\nModels trained successfully! Ready for predictions.")
    get_prediction()