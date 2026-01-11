# Deep Learning-Based Earthquake Early Warning System with Domain Adaptation

This repository contains the source code developed for the MSc Computer Science
dissertation titled:

**Deep Learning-Based Earthquake Early Warning System with Domain Adaptation**

The project investigates deep learning approaches for earthquake early warning
(EEW), with a focus on improving cross-domain generalization using domain
adaptation techniques.

---

## Project Overview

Earthquake Early Warning systems require rapid and reliable inference from
seismic waveform data. Models trained on data from one geographical region often
perform poorly when applied to data from different regions.

This project addresses this challenge by:
- Training deep learning models using the STEAD dataset
- Evaluating model generalization across multiple seismic datasets
- Applying domain adaptation techniques to regional seismic data
- Demonstrating a prototype real-time EEW system

---

## Datasets

The following datasets are used in this project:

- **STEAD**  
  Primary dataset used for model training and initial evaluation.  
  Contains three-channel seismic waveform data sampled at 100 Hz.

- **USGS**  
  Used for cross-dataset evaluation and regression experiments.

- **Myanmar Regional Dataset**  
  Used as the target domain for domain adaptation and final evaluation.

> **Note:**  
> Datasets are not included in this repository due to their size and licensing
> restrictions. Scripts for downloading and preprocessing the data are provided
> in `src/data_processing/`.

---

## Repository Structure

The repository is organised into directories for data processing, training,
evaluation, and shared utility functions.

---

## Configuration Files

The `config/` directory contains:

- `datasets.yaml` — dataset descriptions and metadata
- `model.yaml` — model architecture and training parameters

These files document the experimental setup used in the dissertation and allow
experiments to be reproduced without modifying source code.

---

## Requirements

The project is implemented in Python.

Install dependencies using:

```bash
pip install -r requirements.txt


## Dissertation Context

This repository is provided to support the reproducibility and transparency of
the MSc Computer Science dissertation submitted to the **University of Wolverhampton**.

The code is intended for academic and research purposes only.

---

## Author

Zin Yin Minn  
MSc Computer Science  
University of Wolverhampton
