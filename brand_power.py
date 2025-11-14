import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV

from model_loader import load_brand_model
from predict import teams_list, team_conferences

brand_model = load_brand_model()

numeric_features = [...]     # paste yours
rivalry_features = [...]     # paste yours

def get_brand_power_rankings(year="All"):
    df = pd.read_csv("viewership_cleaned.csv")

    if year != "All":
        df = df[df["Year"] == int(year)]

    power4_set = set(team_conferences.keys()) | {"Notre Dame"}

    d1 = pd.get_dummies(df["Team 1"])
    d2 = pd.get_dummies(df["Team 2"])
    d = d1.add(d2, fill_value=0)

    valid_counts = d.sum()
    d = d[valid_counts[valid_counts >= 3].index]

    feature_cols = [c for c in df.columns if c in numeric_features + rivalry_features]
    X = pd.concat([df[feature_cols], d], axis=1).fillna(0)
    X = sm.add_constant(X)

    y = np.log(df["Persons 2+"] + 1)

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(X, y)

    params = pd.Series(ridge.coef_, index=X.columns)
    team_coefs = params[d.columns]
    team_coefs = team_coefs[team_coefs.index.isin(power4_set)]

    valid_counts = valid_counts.reindex(team_coefs.index).fillna(0)

    adjusted = team_coefs.copy()
    for t in adjusted.index:
        n = valid_counts[t]
        if n <= 4:
            adjusted[t] *= (n / (n + 5))

    boost_pct = (np.exp(adjusted) - 1) * 100

    df_out = pd.DataFrame({
        "Team": adjusted.index,
        "LiftPct": boost_pct.round(1),
        "GamesUsed": valid_counts.astype(int)
    }).sort_values("LiftPct", ascending=False)

    # return pure JSON
    return df_out.to_dict(orient="records")