# DL-fashion-image-classification

This repository contains a small demo project for image classification on Fashion-MNIST. It includes:

...

## Project overview

The demo app is a Streamlit front-end (no separate API server). It loads the Keras model and exposes a simple UI where you can upload an image (jpg/png) and press "Predict". The app does the necessary preprocessing (resize to 28×28, convert to grayscale, normalize) and displays the predicted class.

This README has been updated to reflect the actual project code (Streamlit app) and the repository files.

## Repository structure

- `app/`
	- `main.py` — Streamlit demo app (uploader + prediction)
	- `requirements.txt` — dependencies for the app (see Dependencies)
	- `Dockerfile` — container image that runs the Streamlit app
	- `model/fashion_mnist_cnn_model.h5` — local copy of the trained model used by the app
	- `config.toml`, `credentials.toml` — Streamlit config files (do not store secrets)
- `model/fashion_mnist_cnn_model.h5` — trained model (duplicate of the model in `app/model/`)
- `testset/` — example images to try with the demo app
- `fashion_model_training.ipynb` — training/experimentation notebook
- `requirements.txt` — (root) can be used for notebook/training dependencies

## Dependencies

Recommended Python: 3.8+ (3.10 used in Dockerfile).

App dependencies (from `app/requirements.txt`):

- numpy
- pandas
- matplotlib
- scikit-learn
- keras
- tensorflow
- streamlit
- pillow

Install into a virtualenv (PowerShell example):

```powershell
# create and activate virtualenv
python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1
# install the Streamlit app dependencies
python -m pip install -r app\\requirements.txt
```

If you plan to run the training notebook, install the packages listed in the root `requirements.txt` as well.

## Run (local)

Start the demo app locally using Streamlit:

```powershell
cd app
streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```

Open http://localhost:8501 in your browser. The page lets you upload an image and click "Predict" — the predicted label is shown on the page.

## Run (Docker)

The `app/Dockerfile` is set up to run the Streamlit app. From the `app/` directory build and run the container (PowerShell):

```powershell
cd app
docker build -t fashion-mnist-streamlit .
# map Streamlit's default port 8501 on the host
docker run --rm -p 8501:8501 fashion-mnist-streamlit
```

Notes about the Dockerfile:

- The Dockerfile uses `python:3.10-slim` and installs `-r requirements.txt`.
- The Dockerfile's `EXPOSE 80` is present in the file but Streamlit serves on port 8501 by default; the `docker run -p 8501:8501` mapping above is recommended.

## How the app works

Key behavior from `app/main.py`:

- The app loads the model with `tf.keras.models.load_model('model/fashion_mnist_cnn_model.h5')` on startup.
- Uploaded images are resized to 28×28, converted to grayscale (`convert('L')`), normalized by dividing by 255.0, and reshaped with a batch dimension before being fed to the model.
- Predictions are shown on the Streamlit page. The UI provides an upload control and a "Predict" button.

If you modify the model or move it to a different path, update the path used in `app/main.py` or ensure the file is available in the container at `app/model/fashion_mnist_cnn_model.h5`.

## Model details and labels

- Format: Keras HDF5 (`.h5`) saved model
- Expected input: 28×28 grayscale images normalized to [0,1] and batched (shape: 1, 28, 28)

Label mapping used by the demo (Fashion-MNIST):

0 — T-shirt/top
1 — Trouser
2 — Pullover
3 — Dress
4 — Coat
5 — Sandal
6 — Shirt
7 — Sneaker
8 — Bag
9 — Ankle boot

## Tests and suggested improvements

Current state:

- There are no automated tests included for the app or model loading.

Suggested small improvements (I can implement any of these):

- Add a simple pytest test that imports the preprocessing function and verifies shapes and normalization.
- Add a smoke test that loads the Keras model and runs a dummy input through predict (fast check).
- Add an example client script (Python) that loads the model directly and prints predictions for the files in `testset/`.
- Add CI to run linting and the test suite on push.

## Notes and next steps

- There are two copies of the model file: `model/fashion_mnist_cnn_model.h5` and `app/model/fashion_mnist_cnn_model.h5`. Keep them in sync or remove duplication to avoid confusion.
- For production use, consider exporting a TensorFlow SavedModel, containerizing with a lightweight ASGI server, or converting to TF-Lite/ONNX for edge deployment.

If you'd like, I can now:

- Add a small pytest test and run it locally
- Add a sample Python script that loads `model/fashion_mnist_cnn_model.h5` and prints predictions for the files in `testset/`
- Add a LICENSE file (MIT recommended)

Tell me which one you want and I'll mark the next todo in-progress and implement it.
# DL-fashion-image-classification

A small project demonstrating training and serving a convolutional neural network for Fashion-MNIST image classification. This repository contains a trained Keras model, a minimal FastAPI app to serve predictions, and notebooks / scripts used during development.

## Table of Contents

...

## Project overview

This project trains (and includes a pre-trained) CNN model for the Fashion-MNIST dataset. It also includes a minimal FastAPI-based inference application that loads the Keras model and exposes a prediction endpoint. The repository is intended as a learning/demo project for image classification and model serving.

Key features:

- Pre-trained Keras HDF5 model: `model/fashion_mnist_cnn_model.h5`
- Minimal FastAPI app in `app/main.py` for serving predictions
- Example test images in `testset/`
- A Jupyter notebook `fashion_model_training.ipynb` used during training and experimentation

## Repository structure

Root layout:

- `fashion_model_training.ipynb` - training and exploration notebook
- `model/` - contains the trained model file(s)
- `app/` - FastAPI application code and Dockerfile
- `testset/` - sample images for quick manual testing
- `requirements.txt` - Python dependencies for the root project (if used)

Files of interest in `app/`:

- `main.py` - FastAPI app that loads the model and exposes a `/predict` endpoint
- `Dockerfile` - container image definition for serving the model
- `config.toml` / `credentials.toml` - optional config files (do not store secrets here)

## Requirements

This project uses Python 3.8+ (recommended 3.9 or 3.10). Primary Python packages include:

- tensorflow / keras (for model loading and inference)
- fastapi (for the API)
- uvicorn (ASGI server)
- pillow (PIL) for image loading

Install the dependencies in a virtual environment. Example (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -r app\requirements.txt
```

If you want to run the Jupyter notebook, also install the notebook requirements listed at the root `requirements.txt`.

## Quickstart (local)

1. Create and activate a virtual environment and install app dependencies (see Requirements).
2. Start the FastAPI app with Uvicorn from the `app` folder:

```powershell
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. Open http://127.0.0.1:8000/docs to view the interactive API docs (Swagger UI).

4. Try prediction with an example image from `testset/` using the Swagger UI or a quick PowerShell request. Example (PowerShell):

```powershell
$img = Get-Content -Path ..\testset\fashion_mnist_1.png -Encoding Byte -ReadCount 0
[System.Net.Http.HttpClient]::new().PostAsync('http://127.0.0.1:8000/predict', (New-Object System.Net.Http.ByteArrayContent($img))) | Select -ExpandProperty Result
```

Note: the exact request format depends on how `app/main.py` implements the upload endpoint (file upload or raw bytes). Use the `/docs` UI to confirm the required payload.

## Running with Docker

The `app/Dockerfile` in this repository builds a container that starts the FastAPI app and serves the model. Build and run with Docker (PowerShell):

```powershell
cd app
docker build -t fashion-mnist-api .
docker run --rm -p 8000:8000 -v ${PWD}:/app -e PORT=8000 fashion-mnist-api
```

Then open http://127.0.0.1:8000/docs.

## Using the model

The trained model file is at `model/fashion_mnist_cnn_model.h5`. The model expects 28x28 grayscale images (Fashion-MNIST format). When sending images to the API ensure the server-side code converts inputs to the proper shape (1, 28, 28, 1) and normalizes pixel values to the same scale used during training (commonly [0,1]).

Label mapping (Fashion-MNIST):

0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot

## Tests and verification

Manual test:

1. Start the app (locally or via Docker).
2. Use `/docs` to upload one of the images in `testset/` and verify the returned prediction label and confidence.

Automated tests: This repo doesn't currently include unit tests for the API or model loading. Suggested additions:

- Add pytest-based tests for `app/main.py` that mock model loading and send test images
- CI pipeline to run tests and linting on push

## Notes, tips and next steps

- If you plan to re-train the model, use `fashion_model_training.ipynb` as a starting point. Save checkpoints and experiment with augmentations.
- For production serving consider converting model to a TensorFlow SavedModel or TF Lite / ONNX for improved portability.
- Secure the API before deploying (authentication, rate-limiting, CORS policies).

## License

This repository doesn't include an explicit license file. Add a LICENSE file (MIT or Apache-2.0 are common choices) if you plan to make the project public.

---

If you'd like, I can also:

- Add a basic example client script to call the API
- Add minimal pytest tests and run them
- Add a short CONTRIBUTING.md and LICENSE file

Tell me which follow-up you'd like and I'll proceed.
# DL-fashion-image-classification