# Cricket Bowling Action Analyzer

A Streamlit app that uses pose estimation and machine learning to flag injury risk in a cricket bowling action from an ordinary video.

I built it after a bowling injury of my own. I used it to find and fix the flaws in my action, and I'm back playing competitively.

## What it does

1. **Upload a video** of a bowler (`.mp4`, `.mov` or `.avi`).
2. **Pose estimation.** MediaPipe Pose tracks the bowler's joints on every 5th frame and measures four angles that matter for injury: elbow, spine, front knee and shoulder.
3. **Clip-level risk.** A random-forest classifier takes the clip's average angles and predicts an overall risk level.
4. **Joint-level risk.** A separate model for each joint labels it (for example *Risky* or *Severe*). When a joint is outside its safe range, the app gives coaching advice based on guidance from Level 2 and Level 3 coaches I've trained with.
5. **Frame-by-frame review.** A slider steps through the analyzed frames with the detected joints drawn on, and there is a rotate button for phone footage.

Re-record after changing your action, and the app shows whether the risk went down.

## How the models were built

- **Features:** joint angles computed from MediaPipe landmarks (`pose_utils.py`).
- **Training data:** I wrote a synthetic data generator that adds realistic variance and noise to bowling actions, because labeled injury data for cricket bowlers barely exists.
- **Overfitting check:** on clean synthetic data the model scored 99.9969% accuracy, which was too good to trust. After adding realistic noise and variance, accuracy settled at about 96%, a result that holds up much better on real footage.
- **Models:** scikit-learn random forests, one for the whole clip and one per joint, with a fitted scaler and label encoders.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface: upload, analysis, risk predictions, advice, frame viewer |
| `pose_utils.py` | MediaPipe pose detection and joint-angle calculation |
| `model_loader.py` | Loads the models; the two large random-forest files are downloaded from Google Drive on first run |
| `*.pkl` | Scaler and label encoders used by the models |
| `run_extractor.py` | Exports per-frame pose angles from a video to CSV |
| `test_app.py` | Minimal page that checks Streamlit is installed correctly |

## Run it locally

Use **Python 3.10**. MediaPipe is pinned to 0.10.9, the version the app was built and tested with.

```bash
git clone https://github.com/Anishatl/bowling-streamlit-app.git
cd bowling-streamlit-app
pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads the two model files (about a minute) into `downloaded_models/`. After that they load from the cache.

You can also open the repo in **GitHub Codespaces**: the dev container installs everything and starts the app.

## Tech

Python · MediaPipe · OpenCV · scikit-learn · pandas · NumPy · Streamlit

## Author

**Anish Bhatia**, Computer Engineering at Georgia Tech
[anishbhatia.me](https://anishbhatia.me) · [LinkedIn](https://www.linkedin.com/in/anish-bhatia-733184319)
