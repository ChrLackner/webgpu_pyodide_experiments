# NGSolve with Pyodide and WebGPU

This repository contains an example setup for using NGSolve with Pyodide and WebGPU. Follow the instructions below to set up and run the project, either as a standalone HTML server or within a Jupyter Notebook.

## Setup

To get started, download and extract the required NGSolve Pyodide package:

```bash
wget https://ngsolve.org/files/ngsolve_pyodide_0.26.2.tar.bz2
tar xvf ngsolve_pyodide_0.26.2.tar.bz2
```

## Running as a Standalone HTML Server
To serve the project locally, use Python's built-in HTTP server:

```bash
python -m http.server
```
Then open your browser and navigate to `http://localhost:8000`.

## Running in Jupyter Notebook
To run the project in a Jupyter Notebook, first launch Jupyter:

```bash
jupyter notebook
```
Open the `webgpu.ipynb` file within the notebook interface and execute the cells to interact with WebGPU and Pyodide.

Important: In Jupyter Notebook, running the same cell multiple times can lead to memory issues. Always clear the notebook, then save and reload the page before re-executing the code after making changes.

