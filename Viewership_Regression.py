import pandas as pd
import statsmodels.api as sm
import joblib
import numpy as np
from sklearn.linear_model import RidgeCV

# ======================================================
# 📅 LOAD VIEWERSHIP DATA + FIX YEARS
# ======================================================
df = pd.read_csv("CollegeFootballViewershipWithSpreads.csv", low_memory=False)
df = df[df["Persons 2+"].notna() & (df["Persons 2+"] > 0)]

# ======================================================
# 🕒 CREATE TIME VARIABLES
# ======================================================
def parse_time_slot(t):
    if pd.isna(t) or not isinstance(t, str):
        return None
    try:
        t = t.strip().lower()
        hour, minute = t.replace("a", "").replace("p", "").split(":")
        hour, minute = int(hour), int(minute)
        if "p" in t and hour != 12:
            hour += 12
        if "a" in t and hour == 12:
            hour = 0
        return hour + minute / 60.0
    except:
        return None

df["Time_float"] = df["Time Slot"].apply(parse_time_slot)
df["Monday"] = (df["DoW"] == "Mon").astype(int)
df["Weekday"] = df["DoW"].isin(["Tue", "Wed", "Thu"]).astype(int)
df["Friday"] = (df["DoW"] == "Fri").astype(int)

df["Sat Early"] = ((df["DoW"] == "Sat") &
                   (df["Time_float"].between(11.0, 14.0, inclusive="left"))).astype(int)
df["Sat Mid"] = ((df["DoW"] == "Sat") &
                 (df["Time_float"].between(14.5, 18.5, inclusive="left"))).astype(int)
df["Sat Late"] = ((df["DoW"] == "Sat") &
                  (df["Time_float"].between(21.5, 23.5, inclusive="left"))).astype(int)

# ======================================================
# 📆 YEAR PARSING
# ======================================================
possible_date_cols = [c for c in df.columns if "Date" in c or "date" in c]
if possible_date_cols:
    date_col = possible_date_cols[0]
    df["ParsedDate"] = pd.to_datetime(df[date_col], errors="coerce", format="%m/%d/%y")
    df["Year"] = df["ParsedDate"].dt.year
else:
    raise ValueError("No date column found — please ensure your CSV has a 'Date' column.")

# ======================================================
# 📊 NIELSEN METHODOLOGY DUMMY (pre-2025)
# ======================================================
df["OldNielsenSystem"] = (df["Year"] < 2025).astype(int)

# ======================================================
# 📺 YOUTUBE TV vs DISNEY FEUD DUMMIES (starting 10/30/25)
# ======================================================
feud_start = pd.Timestamp("2025-10-30")
feud_end = None
disney_networks = ["ABC", "ESPN", "ESPN2", "ESPNU"]

if feud_end:
    feud_mask = (df["ParsedDate"] >= feud_start) & (df["ParsedDate"] <= feud_end)
else:
    feud_mask = (df["ParsedDate"] >= feud_start)

for net in disney_networks:
    df[f"YTTV_{net}"] = ((feud_mask) & (df["Station"] == net)).astype(int)

df["YTTV_Other"] = ((feud_mask) & ~df["Station"].isin(disney_networks)).astype(int)

# ======================================================
# 🧢 DEION ERA DUMMY
# ======================================================
df["DeionEra"] = (
    ((df["Team 1"] == "Colorado") | (df["Team 2"] == "Colorado")) &
    (df["Year"].isin([2023, 2024]))
).astype(int)

# ======================================================
# 🔥 RIVALRY VARIABLES
# ======================================================
rivalries = {
    "Michigan_OhioSt": ("Michigan", "Ohio St."), "Texas_Oklahoma": ("Texas", "Oklahoma"),
    "Alabama_Auburn": ("Alabama", "Auburn"), "Georgia_Florida": ("Georgia", "Florida"),
    "NotreDame_USC": ("Notre Dame", "USC"), "Florida_Tennessee": ("Florida", "Tennessee"),
    "Oregon_Washington": ("Oregon", "Washington"), "BYU_Utah": ("BYU", "Utah"),
    "Iowa_IowaSt": ("Iowa", "Iowa St."), "OleMiss_MississippiSt": ("Mississippi", "Mississippi St."),
    "Clemson_SouthCarolina": ("Clemson", "South Carolina"), "Arizona_ArizonaSt": ("Arizona", "Arizona St."),
    "Miami_FloridaSt": ("Miami", "Florida St."), "Texas_TexasA&M": ("Texas", "Texas A&M"),
    "Oregon_OregonSt": ("Oregon", "Oregon St."), "USC_UCLA": ("USC", "UCLA"),
    "Louisville_Kentucky": ("Louisville", "Kentucky"), "Washington_WashingtonSt": ("Washington", "Washington St."),
    "Kansas_KansasSt": ("Kansas", "Kansas St."), "Minnesota_Wisconsin": ("Minnesota", "Wisconsin"),
    "Army_Navy": ("Army", "Navy"), "OhioSt_PennSt": ("Ohio St.", "Penn St."),
    "Alabama_LSU": ("Alabama", "LSU")
}
for name, (t1, t2) in rivalries.items():
    df[name] = (((df["Team 1"] == t1) & (df["Team 2"] == t2)) |
                ((df["Team 1"] == t2) & (df["Team 2"] == t1))).astype(int)
rivalry_features = list(rivalries.keys())

# ======================================================
# 🏆 CONFERENCE POSTSEASON IMPLICATION VARIABLES
# ======================================================
df["BothRanked"] = ((df["Top 10 Rankings"] + df["25-11 Rankings"]) >= 2).astype(int)
df["SEC_PostseasonImplications"] = (df["BothRanked"] & (df["SEC"] == 2)).astype(int)
df["Big10_PostseasonImplications"] = (df["BothRanked"] & (df["Big 10"] == 2)).astype(int)
df["ACC_PostseasonImplications"] = (df["BothRanked"] & (df["ACC"] == 2)).astype(int)
df["Big12_PostseasonImplications"] = (df["BothRanked"] & (df["Big 12"] == 2)).astype(int)

# ======================================================
# 🧩 OHIO STATE × BTN INTERACTION
# ======================================================
df["OhioSt_BTN"] = (
    ((df["Team 1"] == "Ohio St.") | (df["Team 2"] == "Ohio St.")) &
    (df["BTN"] == 1)
).astype(int)

# ======================================================
# 📊 BASE FEATURES
# ======================================================
numeric_features = [
    "Spread", "Competing Tier 1",
    "FOX", "ESPN2", "FS1", "NBC", "CBS", "ABC", "CW", "ESPNU", "FS2", "NFLN", "BTN", "ESPNNEWS",
    "OhioSt_BTN", "Conf Champ", "Sun", "Monday", "Weekday", "Friday", "Sat Early", "Sat Mid", "Sat Late",
    "Top 10 Rankings", "25-11 Rankings", "DeionEra",
    "SEC_PostseasonImplications", "Big10_PostseasonImplications",
    "Big12_PostseasonImplications", "ACC_PostseasonImplications",
    "OldNielsenSystem", "YTTV_ABC", "YTTV_ESPN"
]

# ======================================================
# 🧮 TEAM DUMMIES
# ======================================================
team_dummies_1 = pd.get_dummies(df["Team 1"])
team_dummies_2 = pd.get_dummies(df["Team 2"])
team_dummies = team_dummies_1.add(team_dummies_2, fill_value=0)
team_counts = team_dummies.sum()
valid_teams = team_counts[team_counts >= 5].index
team_dummies = team_dummies[valid_teams]

# ======================================================
# ⚙️ COMBINE BASE MODEL
# ======================================================
X = pd.concat([df[numeric_features + rivalry_features], team_dummies], axis=1)
X = sm.add_constant(X)
y_raw = df["Persons 2+"].astype(float)
df["ln_viewers"] = np.log(df["Persons 2+"] + 1)  # log-transform
y = df["ln_viewers"]

if len(team_dummies.columns) > 0:
    baseline_team = team_dummies.columns[0]
    X = X.drop(columns=[baseline_team])
    print(f"\nℹ️ Dropped {baseline_team} as baseline team to avoid multicollinearity.")

# ======================================================
# 🧮 RUN FULL LOG MODEL
# ======================================================
print("\n============================================================")
print("🏈 RUNNING FULL LOG-TRANSFORMED MODEL (includes OhioSt_BTN interaction)")
print("============================================================")

X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
y = pd.to_numeric(y, errors="coerce").fillna(0).astype(float)

model_full = sm.OLS(y, X).fit()
smearing_factor = np.mean(np.exp(model_full.resid))  # correction for bias
print(model_full.summary())
print(f"\n✅ Smearing correction factor = {smearing_factor:.4f}")

# ======================================================
# 💾 EXPORT CLEANED DATASET
# ======================================================
df.to_csv("viewership_cleaned.csv", index=False)
print("✅ Saved viewership_cleaned.csv for dashboard year filtering.")

# ======================================================
# 🏆 POWER 4 BRAND MODEL (RidgeCV)
# ======================================================
print("\n============================================================")
print("🏆 RUNNING POWER 4 BRAND MODEL (year-aware, adds Pac-12 ≤2023)")
print("============================================================")

PAC12_TEAMS = {
    "Washington", "Oregon", "Arizona", "Oregon St.", "USC", "Utah", "UCLA",
    "California", "Washington St.", "Arizona St.", "Stanford", "Colorado"
}

def is_power4_or_notre_dame(row):
    year = row["Year"]
    team1, team2 = row["Team 1"], row["Team 2"]
    p4_cols = ["SEC", "Big 10", "Big 12", "ACC"]
    power4_count = row[p4_cols].sum()
    if year <= 2023:
        if team1 in PAC12_TEAMS:
            power4_count += 1
        if team2 in PAC12_TEAMS:
            power4_count += 1
    if power4_count >= 2:
        return True
    if "Notre Dame" in [team1, team2] and power4_count >= 1:
        return True
    return False

df_p4 = df[df.apply(is_power4_or_notre_dame, axis=1)].copy()

team_dummies_1_p4 = pd.get_dummies(df_p4["Team 1"])
team_dummies_2_p4 = pd.get_dummies(df_p4["Team 2"])
team_dummies_p4 = team_dummies_1_p4.add(team_dummies_2_p4, fill_value=0)
team_counts_p4 = team_dummies_p4.sum()
valid_p4 = team_counts_p4[team_counts_p4 >= 3].index
team_dummies_p4 = team_dummies_p4[valid_p4]

X_p4 = pd.concat(
    [df_p4[[c for c in numeric_features if c != "DeionEra"] + rivalry_features], team_dummies_p4],
    axis=1
)
X_p4 = sm.add_constant(X_p4)
X_p4["Baseline"] = 0
y_p4 = np.log(df_p4["Persons 2+"] + 1)  # log-scale for Ridge too

X_p4 = X_p4.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
y_p4 = pd.to_numeric(y_p4, errors="coerce").fillna(0).astype(float)
X_p4 = X_p4.drop(columns=["Baseline"])
print("ℹ️ Dropped artificial 'Baseline' column to avoid multicollinearity (kept Alabama).")

ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
ridge.fit(X_p4, y_p4)
ridge_params = pd.Series(ridge.coef_, index=X_p4.columns)
print(f"✅ RidgeCV completed with best alpha = {ridge.alpha_:.3f}")

valid_team_cols = [t for t in team_dummies_p4.columns if t in ridge_params.index]
team_coefs = ridge_params[valid_team_cols]
team_counts_aligned = team_dummies_p4.sum().reindex(team_coefs.index).fillna(0)
adjusted_team_coefs = team_coefs.copy()

print("\n🔧 Applying sample-size shrinkage to Power 4 teams with ≤5 games...")
for team in team_coefs.index:
    games = team_counts_aligned[team]
    if games <= 5:
        factor = games / (games + 5)
        adjusted_team_coefs[team] = team_coefs[team] * factor
        print(f"🔻 {team}: {games} games → scaled by {factor:.2f}")

brand_rankings = (
    pd.DataFrame({
        "Team": team_coefs.index,
        "Raw Coefficient": team_coefs.values,
        "Adjusted (Shrinkage)": adjusted_team_coefs.values,
        "Games Used": team_counts_aligned.values
    })
    .sort_values("Adjusted (Shrinkage)", ascending=False)
    .reset_index(drop=True)
)
print("\n🏆 TOP 25 POWER 4 BRAND RANKINGS:")
print(brand_rankings.head(25).to_string(index=False))

# ======================================================
# 💾 SAVE MODELS
# ======================================================
model_metadata = {
    "model": model_full,
    "team_counts": team_counts.to_dict(),
    "smearing_factor": smearing_factor
}
joblib.dump(model_metadata, "viewership_model_log.joblib")

brand_model_metadata = {
    "model": ridge,
    "params": ridge_params.to_dict(),
    "team_counts": team_counts_p4.to_dict()
}
joblib.dump(brand_model_metadata, "brand_model.joblib")

brand_rankings.to_csv("brand_rankings.csv", index=False)

print("\n✅ Saved:")
print(" - viewership_model_log.joblib (for predictions, log-scale)")
print(" - brand_model.joblib (for dashboard)")
print(" - brand_rankings.csv (Power 4 adjusted rankings)")

# ======================================================
# 🔥 RIVALRY EFFECTS
# ======================================================
rivalry_coefs = model_full.params[rivalry_features].sort_values(ascending=False)
print("\n🔥 Rivalry Effects (Full Model):")
print(rivalry_coefs)