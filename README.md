# 🧠 Fetal Brain MRI Image Quality Assessment (IQA)

A deep learning project for evaluating the quality of fetal brain MRI scans. This repository provides tools for training models, analyzing results, and generating figures for research and thesis work.

---

## 🚀 Overview

This project focuses on:

* 📊 Assessing MRI image quality using deep learning
* 🧪 Running controlled experiments with reproducible scripts
* 📈 Analyzing results with statistical rigor
* 📝 Generating publication-ready figures

---

## 📁 Repository Structure

```
.
├── src/                    # Core source code
│   ├── train.py            # Main training entry point
│   ├── model.py            # Model definitions
│   ├── data.py             # Data loading utilities
│   ├── brain_transforms.py # Preprocessing transforms
│   ├── train_setup.py      # Training configuration
│   ├── evaluate.py         # Evaluation logic
│   ├── display_utils.py    # Visualization helpers
│   ├── parse_args.py       # CLI argument definitions
│   │
│   ├── results/            # Result analysis notebooks
│   └── label_sessions/     # ⚠️ Experimental / messy labeling code
│
├── train_scripts/          # Shell scripts for experiments
├── thesis_figs/            # Figure generation for thesis
├── old_code/               # Deprecated / unused code
│
├── environment.yaml        # Conda environment config
├── .env                    # Environment variables (user-defined)
├── .gitignore              # Git ignore rules
└── README.md               # Documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:MedicalVisionGroup/FetalBrainIQA.git
cd FetalBrainIQA
```

---

### 2. Create Environment Variables

Create a `.env` file in the root directory:

```bash
touch .env
```

Add the following variables:

#### 📂 DATA_PATH

Path to the master CSV file containing all dataset entries (labeled and unlabeled).

Example:

```
/data/vision/polina/users/marcusbl/bin_class/label_sessions_data/label_session_3-11/final.csv
```

This file references data stored in locations such as:

* `/data/vision/polina/projects/fetal/common-data/BRAIN-IQA`
* `/data/vision/polina/users/marcusbl/data/`

#### 📁 OUTPUT_DIR_ROOT

Directory where all experiment outputs will be saved.

---

### 3. Install Dependencies

```bash
conda env create -f environment.yml
conda activate bin_class
```

---

## 🏃 Running Experiments

Run training with:

```bash
python -m src.train [OPTIONS]
```

### 🔍 Helpful Resources

* `train_scripts/` → Example experiment configurations
* `src/parse_args.py` → Full list of available parameters

### 📦 Output Behavior

* Results are stored in `OUTPUT_DIR_ROOT`
* Each experiment run is saved in a separate folder
* Multiple runs enable statistical comparison

---

## 📊 Analyzing Results

Use the provided Jupyter notebook:

```
src/results/results_metrics.ipynb
```

### Steps:

1. Set the following variables:

   * `output_dir`
   * `model_name`
   * `metrics`
2. Run all cells

### ✨ Features:

* Aggregates results across runs
* Displays metrics clearly
* Computes statistical significance

---

## 📈 Generating Thesis Figures

To generate figures for reports or publications:

1. Open notebooks in:

   ```
   thesis_figs/
   ```
2. Update any required paths
3. Run all cells

### 📁 Output:

* Saved to: `thesis_figs/figs`
* Ready for direct use in LaTeX documents

---

## ⚠️ Notes & Caveats

* `label_sessions/` contains unstructured, experimental code
* `old_code/` is deprecated and may be broken
* Always verify `.env` paths before running experiments

---

## 🧩 Tips for Use

* Start with provided shell scripts for reproducibility
* Run multiple seeds for reliable results
* Use notebooks for deeper analysis instead of raw logs

---

## 📬 Contact / Contributions

For questions or improvements, feel free to open an issue or contribute to the repository.

---

## ⭐ Acknowledgment

Developed as part of research in fetal brain MRI quality assessment using deep learning.
