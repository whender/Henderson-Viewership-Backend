"""Build the user-approved Week 1 artifact from the sibling research workspace."""
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'RatingsAndRegression'))
from week1_major_day_interactions import main,preview

if __name__=='__main__':
    preview(*main(),artifact_path=BASE/'week1_major_days.joblib')
