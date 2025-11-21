# LLM PROMPTS AND VALIDATION NOTES

## 1) Frontend – Project setup (React + Vite + TypeScript)

**Prompt**

> We’re building a small frontend for a machine vision project.  
> Please scaffold a React + Vite + TypeScript project structure with:
> - `src/main.tsx` as the React entry point
> - `src/App.tsx` as the root component
> - A simple layout with a header, main content area, and footer
> - Basic routing prepared with `react-router-dom`, with routes for:
    >   - `/` – upload page for images or video frames
>   - `/results` – page that displays model inference results
      > Use functional components and React Hooks. Include minimal but clear TypeScript types for props and state.

**Validation**  
We compared the generated file structure with the official Vite + React + TS template and confirmed imports/entrypoints matched the documented setup. Then we ran `npm run dev` to ensure the app started correctly without TypeScript or runtime errors and verified navigation between `/` and `/results` worked in the browser.


## 2) Frontend – Routing and API integration

**Prompt**

> In a React + Vite + TypeScript app using `react-router-dom`, create routing for:
> - `/` – a page with a file upload form to send an image to a backend API (`POST /api/predict`)
> - `/results` – a page that reads prediction results from React context or `location.state` and displays them
    > Implement a simple `ApiClient` helper that sends a multipart/form-data request with the file and returns JSON predictions.
    > Show how to handle loading and error states in TypeScript.

**Validation**  
We manually tested the flow in the browser by uploading valid and invalid files and inspected network requests in DevTools to ensure the frontend hit the correct endpoint and sent `multipart/form-data`. We also added simple unit checks for the API helper (mocking `fetch`) and verified correct behavior for success, error, and network-failure cases.


## 3) Frontend – Simple results view and state management

**Prompt**

> Write a TypeScript React component called `ResultsView` that:
> - Accepts a `predictions` prop: an array of objects with `label: string`, `score: number`
> - Renders a table of predictions sorted by descending score
> - Highlights the top prediction
    > Include basic prop typing and default handling when no predictions are available.

**Validation**  
We wrote a small storybook-like test setup and passed different `predictions` arrays (including empty and malformed data) to verify the component rendered gracefully. We also checked TypeScript for type errors and confirmed sorting/highlighting matched expectations visually.


## 4) Backend – FastAPI app and routing skeleton

**Prompt**

> We have a Python backend for a machine vision task.  
> Please write a minimal FastAPI application with:
> - A root health-check endpoint `GET /health` returning `{"status": "ok"}`
> - An inference endpoint `POST /predict` that accepts an uploaded image file (`UploadFile`) and returns a dummy JSON response with `{"label": "example", "score": 0.99}` for now.
    > Structure the code so that model loading and prediction logic can later be extracted into a separate `services/model.py` module.

**Validation**  
We started the app locally (`uvicorn`) and called `/health` and `/predict` with `curl` and Python `requests` to confirm the expected JSON structure and HTTP status codes. We also ran a quick integration test using FastAPI’s `TestClient` to ensure file upload handling worked as expected.


## 5) Backend – Model service abstraction

**Prompt**

> Create a Python module `services/model.py` that defines:
> - A `ModelService` class with methods `load_model()` and `predict(image_bytes: bytes) -> dict`
> - A simple in-memory singleton pattern so the model loads once and can be reused across requests
    > For now, implement `predict` to return a mock result like `{"label": "mock_object", "score": 0.95}`, but keep the interface clean so it can be swapped with a real model later.

**Validation**  
We instantiated `ModelService` in a Python REPL and called `predict` with dummy image bytes to check that the interface and return format were stable. Later, when integrating the real model, we reused the same interface and verified that FastAPI responses did not need to change, indicating a correct abstraction boundary.


## 6) Backend – Connecting FastAPI endpoint with model service

**Prompt**

> Update the FastAPI `POST /predict` endpoint so that it:
> - Reads the uploaded image file as bytes
> - Passes the bytes into `ModelService.predict()`
> - Returns the prediction dict as JSON
    > Handle basic error cases: missing file, unsupported content type, and generic server error, returning appropriate HTTP status codes.

**Validation**  
We wrote small automated tests using FastAPI’s `TestClient` to send: (1) a valid image file, (2) no file, and (3) a text file, verifying the status codes and response bodies matched the expected error handling. We also inspected logs to confirm that exceptions were caught and meaningful messages were returned.


## 7) Backend – CORS configuration for frontend integration

**Prompt**

> Configure CORS in a FastAPI application so that a Vite dev server running at `http://localhost:5173` can call the API.
> Show how to:
> - Add `CORSMiddleware` with allowed origins, methods, and headers
> - Keep the configuration easy to extend later for production domains.

**Validation**  
We attempted to call the backend from the running Vite frontend and confirmed requests no longer failed with CORS errors in the browser console. We also briefly tightened the allowed origins list to ensure that invalid origins were blocked as expected.


## 8) Shared – Simple Dockerfile / environment configuration

**Prompt**

> Provide a minimal Dockerfile for the Python FastAPI backend that:
> - Uses a slim Python base image
> - Installs dependencies from `requirements.txt`
> - Exposes port 8000
> - Runs the app with `uvicorn main:app --host 0.0.0.0 --port 8000`
    > Optimize a bit for smaller image size (e.g., using `--no-cache-dir` and a non-root user if simple).

**Validation**  
We built and ran the Docker image locally, then used `curl` from the host to hit `/health` and `/predict`, confirming that the application behaved the same way as in the local (non-Docker) environment. We also inspected the image size and layers to ensure there were no obvious issues like unused build steps.

## 9) Machine Vision model

**Prompt**

> Create a python program that will analyze a car dashcam video, the main purpose is to detect **dangerous events**
> The program will detect sidelines of the lane the car is currently in a monitor its velocity (and velocity of other objects) and an area before the car. When an object enters the area before the car and their relative velocity is greater than some threshold (risk of colission), the program shall report a **dangerous event**
> Use modern python, runnable in google colab (output only one code block) with modern packages/algorithms for year 2025
> # Features
> The program will have the following features
> - calculation of velocities (use relevant ultralytics https://docs.ultralytics.com/guides/speed-estimation/)
> - object detection, tracking and logging counts (https://docs.ultralytics.com/guides/instance-segmentation-and-tracking/)
> - dangerous event identification (use relative velocities and "alarm area" polygon before the car)
> - Time profiling (detected objects per second or per 5 s).
> - Spatial pattern (heatmap of detection centers or occupancy by region).
> - DD-MM-YYYY hh:mm:ss timestamp in top right corner (starting at provided start point)
> - the vehicle's speed should be under the timestamp in top right corner
> # Guidelines
> - allow configuration via constants at the top of the script
> - default video location is ./video.mp4
> - default output dir is ./
> - default ultralytics model is yolo11x
> - print progress (ex. each 100 frames + percentage)
> - clearly mark when the "dangerous" event happens in the video, don't allow multiple dangerous events in the span of x seconds
> - analyse only each n-th frame by the vision model (but add all the unanalyzed frames to the resulting video, overlay them with the data from the latest frame)
> - use cuda for analysis if possible

**Validation**  
We ran the script on short dashcam clips containing both normal driving and synthetic near-collision scenarios to confirm that dangerous events were only triggered when objects entered the alarm area with sufficiently high relative closing speed. We visually inspected the output video to verify overlay correctness (lanes, timestamps, vehicle speed, and danger markers) and cross-checked the logged counts, time profiling, and heatmaps against a few manually counted frames. We also tested both CPU-only and CUDA-enabled environments in Colab to ensure the program fell back gracefully when GPU was unavailable. 