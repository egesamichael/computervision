# Coffee Disease Intelligence (VGG16 Demo)

This project packages a pretrained `VGG16` model into a Flask web application that mimics a coffee disease screening workflow. Users upload images of coffee leaves or berries, the server runs ImageNet inference, and the UI presents a diagnosis summary, recommendations, and a saved analysis log.

## Features

- Upload field photos of coffee leaves or berries
- Model-driven detection with top predictions and confidence scores
- Diagnosis summary with control/treatment recommendations
- Optional notes and location metadata captured per analysis
- Recent analysis history stored locally in SQLite
- Full history page with CSV export

## Model Choice

This app is wired for a coffee disease classifier based on **EfficientNetB0**. The model is expected at:

- `/Users/praisewebsolutions/computervision/models/coffee_disease_efficientnetb0.keras`

The label order is:

1. Healthy
2. Coffee Leaf Rust
3. Cercospora Leaf Spot
4. Phoma Leaf Spot
5. Coffee Berry Disease

If the model file is missing, the app automatically falls back to a generic ImageNet `VGG16` model and shows a notice in the UI.

## Important Note

Replace the model file with your trained coffee disease weights for production use. The fallback ImageNet model is only for demo purposes.

## Teachable Machine Model Conversion (Recommended)

Some Teachable Machine exports are legacy `.h5` models that do not load cleanly in
TensorFlow 2.21 / Keras 3. Convert once to a `.keras` file using a Python 3.11
environment with TensorFlow 2.13.x:

```bash
python3.11 -m venv /tmp/tm-convert
source /tmp/tm-convert/bin/activate
pip install --upgrade pip
pip install "tensorflow==2.13.*"
python scripts/convert_teachable_model.py \
  --in models/keras_model.h5 \
  --out models/coffee_disease_compatible.keras
```

Then set:

```bash
export COFFEE_MODEL_PATH="/absolute/path/to/coffee_disease_compatible.keras"
```

## Project Layout

```text
.
├── app
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── recommendations.py
│   ├── routes.py
│   ├── storage.py
│   ├── static/css/styles.css
│   └── templates/
├── Dockerfile
├── Procfile
├── requirements.txt
├── runtime.txt
└── wsgi.py
```

## Local Setup

The most important setup detail is the Python version.

- `Python 3.14` will not work for this project because TensorFlow does not publish compatible wheels for it yet.
- Use `Python 3.11` or `Python 3.12`.
- On macOS, install `tensorflow`, not `tensorflow-cpu`.

If your machine only has Python 3.14 right now, install Python 3.12 first, then create a fresh virtual environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m flask --app wsgi run --debug
```

Open `http://127.0.0.1:5000`.

If `python3.12` is not available yet, install Python 3.12 from python.org or Homebrew, then rerun the commands above.

## Windows Setup (PowerShell)

Run these commands from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m flask --app wsgi run --debug
```

Open `http://127.0.0.1:5000`.

Optional model path override:

```powershell
$env:COFFEE_MODEL_PATH="C:\\full\\path\\to\\coffee_disease_efficientnetb0.keras"
$env:COFFEE_MODEL_INPUT_SIZE="224"
```

## Windows Setup (Command Prompt)

Run these commands from the project root:

```cmd
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m flask --app wsgi run --debug
```

Open `http://127.0.0.1:5000`.

Optional model path override:

```cmd
set COFFEE_MODEL_PATH=C:\full\path\to\coffee_disease_efficientnetb0.keras
set COFFEE_MODEL_INPUT_SIZE=224
```

## Docker Run

```bash
docker build -t vgg16-flask-app .
docker run --rm -p 8000:8000 -e PORT=8000 vgg16-flask-app
```

Open `http://127.0.0.1:8000`.

## Deployment Notes

- The `Procfile` supports platforms that launch `gunicorn` directly.
- The `Dockerfile` pre-downloads VGG16 ImageNet weights during image build so the first request is fast in production.
- For a platform like Render, Railway, or Fly.io, deploy from this directory and use the included `Dockerfile`.
- Set `SECRET_KEY` in production instead of relying on the default placeholder value.
- The included [runtime.txt](/Users/praisewebsolutions/computervision/runtime.txt) targets Python `3.11.10` for hosted deployments.

## How Prediction Works

1. Flask receives the uploaded file and metadata.
2. Pillow converts the image to RGB and resizes it to `224 x 224`.
3. TensorFlow applies `preprocess_input` for VGG16.
4. The model returns ImageNet probabilities, and the app displays diagnosis, recommendations, and a saved history entry.

## History & Export

- Visit `http://127.0.0.1:5000/history` for the full analysis log.
- Download the CSV export from `http://127.0.0.1:5000/history.csv`.

## Configuration

You can change the model path or input size with environment variables:

```bash
export COFFEE_MODEL_PATH="/absolute/path/to/your_model.keras"
export COFFEE_MODEL_INPUT_SIZE=224
```
