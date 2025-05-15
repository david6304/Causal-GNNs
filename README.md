# Part II Project - Causal GNNs

This project implements and evaluates causal GNNs against non-causal baseline on synthetic and real world medical data. 

## Table of Contents

1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Usage](#usage)
4. [License](#license)

---

## Requirements

- Python 3.12

## Installation

1. Clone the repository
2. Create a virtual environment (Optional):

```bash
conda create -n project_name python=3.12
conda activate project_name
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To reproduce any of the results from the project, go to the experiments folder and run the relevant script. The real world data used in this project is sensitive and so cannot be accessed. To reproduce the synthetic experiments first go to src/data_preprocessing/synthetic_data.py and run that script to generate the synthetic data files.  

## License

MIT License - see LICENSE file for more information.
