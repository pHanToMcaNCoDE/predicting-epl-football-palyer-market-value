import re
import time
import requests
import unicodedata
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#######################################################################################################################################
# Global variable(s)
#######################################################################################################################################

# The last 3 digits of my Student ID (R00277181).

random_for_all = 181

# Random Forest

rf_param_grid = {
    "model__n_estimators": [200, 500, 800],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

# Light GBM

lgbm_param_grid = {
    "model__n_estimators": [200, 500, 800],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__num_leaves": [15, 31, 63]
}

# XGBoost

xgb_param_grid = {
    "model__n_estimators": [200, 500, 800],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5, 7]
}

#######################################################################################################################################
# Raw data
#######################################################################################################################################

PLAYER_NAME_MAPPING = {
    "savio": "savinho",
    "kostas tsimikas": "konstantinos tsimikas",
    "son heung min": "heung min son",
    "emi buendia": "emiliano buendia",
    "andy irving": "andrew irving",
    "gabriel magalhaes": "gabriel",
    "igor": "igor julio",
    "toti gomes": "toti",
    "nico oreilly": "nico oreilly",
    "victor bernth kristiansen": "victor kristiansen",
    "william smallbone": "will smallbone",
    "yehor yarmoliuk": "yegor yarmolyuk",
    "yunus emre konak": "yunus konak"
}

#######################################################################################################################################
# REUSABLE FUNCTIONS
#######################################################################################################################################

# All playing positions

current_positions = [
    "Goalkeeper",
    "Centre-Back",
    "Left-Back",
    "Right-Back",
    "Defensive Midfield",
    "Central Midfield",
    "Attacking Midfield",
    "Right Midfield",
    "Left Winger",
    "Right Winger",
    "Centre-Forward",
    "Second Striker",
    "Left Midfield"
]


# Extract Playing Positions from Player names

def extract_position(text):
    """ A function that extracts a player's position from the Transfermarkt Player column.

        Parameters:
            text (str):
                A string containing both the player's name and position.

        Returns:
            str or None: The identified position if found. Returns None if no known position is detected.

    """

    for pos in current_positions:
        if text.endswith(pos):
            return pos
    return None


# Extract Player Names

def extract_player_name(text, position):
    """A function to extract the player's name from the Transfermarkt Player column.

    Parameters:
        text (str): Original Player column value containing both player name and position.

        position (str): Position extracted from the Player column.

    Returns:
        str: A clean player name with the position removed.
    """

    if position is None:
        return text.strip()

    return text.replace(position, "").strip()


# Convert Market Value from characters to numbers

def convert_market_value(value):
    """To convert Transfermarkt market values from strings into numeric values.

    Parameters:
        value (str): Player Market value string from Transfermarkt

    Returns:
        float: Market value in euros.
    """

    value = value.replace("€", "")

    if "m" in value:
        return float(value.replace("m", "")) * 1_000_000

    elif "k" in value:
        return float(value.replace("k", "")) * 1_000

    return None


# Clean transfermarket data for every squad/clubs

def clean_transfermarkt_squad(df, club, season):
    """ Clean and transform a Transfermarkt squad table.

        This function processes a raw Transfermarkt squad DataFrame by:
            - Removing rows with missing market values.
            - Extracting player positions from the Player column.
            - Extracting player names from the Player column.
            - Converting market values from text format into numeric values.
            - Adding club and season identifiers.
            - Returning a simplified dataset suitable for analysis.

        Parameters:

            df (pandas.DataFrame): Raw Transfermarkt squad table.

            club (str): Name of the football club.

            season (str): Season identifier (e.g. '2122', '2223').

        Returns:

            pandas.DataFrame: Cleaned squad dataset containing player name, position, age, market value, club, and season.
    """

    df = df[df["Market value"].notna()].copy()
    df = df[df["Age"].notna()].copy()

    # extract positions

    df["position"] = df["Player"].apply(extract_position)

    # extract player name

    df["player_name"] = df.apply(
        lambda row: extract_player_name(
            row["Player"],
            row["position"]
        ),
        axis=1
    )

    # convert market value

    df["market_value_eur"] = (
        df["Market value"]
        .apply(convert_market_value)
    )

    df["club"] = club
    df["season"] = season

    return df[
        [
            "player_name",
            "position",
            "Age",
            "market_value_eur",
            "club",
            "season"
        ]
    ]

# get clubs for each league

def get_league_clubs(league_url):
    """Extract club names and club IDs from a Transfermarkt league page.

    Parameters:
        league_url : str
            Transfermarkt league URL.

    Returns:
        pd.DataFrame
            DataFrame containing:

            - club_name
            - club_id
            - squad_url
    """

    html = requests.get(
        league_url,
        headers={"User-Agent": "Mozilla/5.0"}
    ).text

    soup = BeautifulSoup(html, "html.parser")

    club_links = []

    for link in soup.find_all("a"):

        href = link.get("href")

        if href and "/kader/verein/" in href:
            club_links.append(href)

    unique_clubs = sorted(set(club_links))

    club_records = []

    for club in unique_clubs:

        club_records.append(
            {
                "club_name": (
                    club.split("/")[1]
                    .replace("-", " ")
                    .replace(" amp ", " and ")
                    .title()
                ),
                "club_id": int(
                    re.search(
                        r"verein/(\d+)",
                        club
                    ).group(1)
                ),
                "squad_url": (
                    "https://www.transfermarkt.com"
                    + club
                )
            }
        )

    return pd.DataFrame(club_records)



# Collect all squads for multiple seasons



def collect_league_squads(clubs_df, season):
    """ Collect and clean squad data for all clubs
    in a league.

    Parameters: 
        clubs_df (pd.DataFrame): Output from get_league_clubs().
        season (str): Season identifier.

    Returns:
        pd.DataFrame: Combined squad data for all clubs.
    """

    all_squads = []

    for _, row in clubs_df.iterrows():

        print("Processing:", row["club_name"])

        for attempt in range(3):

            try:
                tables = pd.read_html(row["squad_url"])

                squad_df = clean_transfermarkt_squad(
                    tables[1],
                    row["club_name"],
                    season
                )

                all_squads.append(squad_df)

                print(f"✓ {row['club_name']}")

                time.sleep(2)
                break

            except Exception as e:

                print(
                    f"Attempt {attempt + 1} failed for "
                    f"{row['club_name']}: {e}"
                )

                time.sleep(5)

                if attempt == 2:
                    print(f"Skipping {row['club_name']}")

    print("Number of squad dataframes:", len(all_squads))

    return pd.concat(
        all_squads,
        ignore_index=True
    )


# Function to remove symbols or alpha-numeric characters in player names

def clean_player_name(name):
    """ Standardise player names for merging datasets.

        Parameters:
            name : str

        Returns:
            str
    """

    if pd.isna(name):
        return name

    # Remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(
        c for c in name
        if not unicodedata.combining(c)
    )

    # Lowercase
    name = name.lower()

    # Replace punctuation with spaces
    name = re.sub(r"[-']", " ", name)

    # Remove non-alphanumeric characters
    name = re.sub(r"[^a-z0-9 ]", "", name)

    # Remove extra spaces
    name = re.sub(r"\s+", " ", name)

    return name.strip()





# Merge the data sources

def merge_fbref_transfermarkt(fbref_df, transfermarkt_df):
    """Merge FBref performance data with transfermarkt market values.

        Returns:
            merge (DataFrame): Merged dataset
    """

    merged = fbref_df.merge(
        transfermarkt_df,
        on=[
            "player_clean",
            "club",
            "season"
        ],
        how="inner"
    )

    return merged




# Fix bundesliga clubs

def fix_bundesliga_league(df):

    df = df.reset_index()

    bundesliga_clubs = [
        "Arminia",
        "Augsburg",
        "Bayern Munich",
        "Bochum",
        "Dortmund",
        "Eintracht Frankfurt",
        "Freiburg",
        "Gladbach",
        "Greuther Fürth",
        "Heidenheim",
        "Hoffenheim",
        "Holstein Kiel",
        "Leverkusen",
        "Mainz 05",
        "RB Leipzig",
        "Schalke 04",
        "St Pauli",
        "Stuttgart",
        "Union Berlin",
        "Werder Bremen",
        "Wolfsburg",
        "Köln",
        "Hertha BSC",
        "Darmstadt 98",
        "Frankfurt"
    ]

    df.loc[
        df["team"].isin(bundesliga_clubs),
        "league"
    ] = "GER-Bundesliga"

    return df

#################################################################################################################################
# TRAINING PIPELINE FUNCTION
#################################################################################################################################

# Random Forest

def run_random_forest(X_train, X_test, y_train, y_test, model_name):
    """
        Train, tune and evaluate a Random Forest model using GridSearchCV with 5-fold cross-validation.
    """

    # Build Random Forest pipeline
    random_forest_pipeline = Pipeline([
        ("model", RandomForestRegressor(
            n_estimators = 500,
            random_state=random_for_all,
            n_jobs=-1
        ))
    ])

    # Hyperparameter tuning using 5-fold cross-validation

    rf_grid_search = GridSearchCV(
        estimator=random_forest_pipeline,
        param_grid=rf_param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )

    # Fit GridSearchCV to find the best Random Forest model
    rf_grid_search.fit(X_train, y_train)
    


    # prediction using test data
    y_pred_rf = rf_grid_search.predict(
        X_test
    )


    # Model Evaluation
    
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"{model_name} Random Forest completed.")
    
    return {
        "Algorithm": "Random Forest",
        "Dataset": model_name,
        "MAE": mae_rf,
        "RMSE": rmse_rf,
        "R2": r2_rf,
        "Best Parameters": rf_grid_search.best_params_,
        "Best Estimator": rf_grid_search.best_estimator_
        #"Predictions": y_pred_rf
    }


    


# Linear Regression

def run_linear_regression(X_train, X_test, y_train, y_test, model_name):
    """ 
        Train and evaluate a Linear Regression model using an 80/20 train-test split.
    """

    # Build Linear Regression pipeline
    linear_regression_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    # Fit model
    linear_regression_pipeline.fit(
        X_train,
        y_train
    )

    # Make predictions
    y_pred_lr = linear_regression_pipeline.predict(
        X_test
    )

    # Model evaluation
    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    rmse_lr = np.sqrt(
        mean_squared_error(y_test, y_pred_lr)
    )

    r2_lr = r2_score(y_test, y_pred_lr)

    print(f"{model_name} Linear Regression completed.")

    return {
        "Algorithm": "Linear Regression",
        "Dataset": model_name,
        "MAE": mae_lr,
        "RMSE": rmse_lr,
        "R2": r2_lr
        #"Predictions": y_pred_lr
    }    


    

    
# Light Gradient Boosting Model

def run_light_gbm(X_train, X_test, y_train, y_test, model_name):
    """
        Train, tune and evaluate a LightGBM model using an 80/20 train-test split.
    """

    # Build LightGBM pipeline
    lgbm_pipeline = Pipeline([
        ("model", LGBMRegressor(
            n_estimators=500,
            random_state=random_for_all,
            n_jobs=-1,
            verbosity=-1
        ))
    ])

    # Hyperparameter tuning using 5-fold cross-validation
    lgbm_grid_search = GridSearchCV(
        estimator=lgbm_pipeline,
        param_grid=lgbm_param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )

    # Fit GridSearchCV to find the best LightGBM
    lgbm_grid_search.fit(X_train, y_train)
    

    # Make predictions on the test set
    y_pred_lgbm = lgbm_grid_search.predict(
        X_test
    )

    # Model Evaluation
    
    mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
    rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
    r2_lgbm = r2_score(y_test, y_pred_lgbm)
    
    print(f"{model_name} LightGBM completed.") 
    
    return {
        "Algorithm": "LightGBM",
        "Dataset": model_name,
        "MAE": mae_lgbm,
        "RMSE": rmse_lgbm,
        "R2": r2_lgbm,
        "Best Parameters": lgbm_grid_search.best_params_,
        "Best Estimator": lgbm_grid_search.best_estimator_
        #"Predictions": y_pred_lgbm
    }





# XGBoost


def run_xgboost(X_train, X_test, y_train, y_test, model_name):
    """
        Train, tune and evaluate an XGBoost model using GridSearchCV with 5-fold cross-validation.
    """

    
    # Build XGBoost pipeline
    xgb_pipeline = Pipeline([
        ("model", XGBRegressor(
            n_estimators=500,
            random_state=random_for_all,
            n_jobs=-1
        ))
    ])

    # Hyperparameter tuning using 5-fold cross-validation

    xgb_grid_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=xgb_param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )

    # Fit GridSearchCV to find the best XGBoost model
    xgb_grid_search.fit(X_train, y_train)
    


    # prediction using test data
    y_pred_xgb = rf_grid_search.predict(
        X_test
    )


    # Model Evaluation
    
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)

    print(f"{model_name} XGBoost completed.")
    
    return {
        "Algorithm": "Random Forest",
        "Dataset": model_name,
        "MAE": mae_xgb,
        "RMSE": rmse_xgb,
        "R2": r2_xgb,
        "Best Parameters": xgb_grid_search.best_params_,
        "Best Estimator": xgb_grid_search.best_estimator_
        #"Predictions": y_pred_xgb
    }

    