## Lyrics Genre Classifier (Spark MLlib + Web UI)

A local web app that classifies pasted song lyrics into **8 music genres** using a Spark MLlib pipeline.  
When you click **Classify**, the UI shows:

- **Predicted genre**
- **Bar chart** of scores for **all 8 classes**
- **Table** of the same scores (so runner-up genres are visible)

### Genres (from the trained model)
The current saved model in `music_genre_model/` contains these labels:

- jazz, blues, indie, rock, country, hip hop, reggae, pop

> Note: Scores are visualized by applying a softmax normalization to the model’s raw margins (this allows a “probability-like” chart even when using margin-based classifiers).

---

## Project structure

- `spark/pyspark_server.py`
  - Starts a SparkSession
  - Loads the trained model from `music_genre_model/`
  - Serves:
    - `GET /api/health` → `ok`
    - `POST /api/predict` → JSON `{ predicted, labels, scores[] }`
  - Serves the frontend from `web/`
- `web/`
  - `index.html` + `styles.css` + `app.js`
  - Uses **Chart.js** to render a **bar chart**
- `run.bat`
  - One command to start the backend and open the browser

---

## Requirements (Windows)

- **Apache Spark** installed (this repo assumes Spark 4.1.1)
- **Python 3.10** available via the Windows `py` launcher:
  - `py -3.10 --version` should work
- Python dependencies installed for Python 3.10:
  - `numpy` (required for `pyspark.ml`)

Install numpy:

```bat
py -3.10 -m pip install numpy
