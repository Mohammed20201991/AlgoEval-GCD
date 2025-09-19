# AlgoEval-GCD
**Intelligent Cloud Resource Usage Potentially to Improve Task Scheduling by the use of Artificial Intelligence**

**Algorithm Evaluation on Google Cluster Data (GCD)**  
This repository contains code and experiments for evaluating different algorithms on the **Google Cluster Data (GCD)** dataset.  
The goal is to analyze datacenter workloads, apply machine learning methods, and benchmark ensemble approaches for workload prediction and anomaly detection.

---

## Dataset: Google Cluster Data (GCD)

We use the **Google Borg cluster traces** ([Google Cluster Data](https://github.com/google/cluster-data)), which capture workload behavior in large-scale datacenters.  
The dataset provides detailed logs about:

- **Jobs and tasks** (scheduling, start/finish, resource usage)  
- **Resource allocation** (CPU, memory, disk I/O, etc.)  
- **Events and constraints** in cluster management  

This dataset enables **research on scheduling, workload prediction, and anomaly detection** in real-world production clusters.

---

## Repository Structure

The repository contains the following Python files:

- **`ai+cloud.py`** → ML/AI-based methods for analyzing cloud workloads.  
- **`googleborg.py`** → Exploring Google Borg trace parsing and analysis.  
- **`google_cluster.py`** → Core data preprocessing and feature engineering from GCD.  
- **`ensemble_with_google.py`** → Ensemble learning techniques applied on cluster data.  

---

## Usage

### Clone Repository
```
git clone https://github.com/your-username/AlgoEval-GCD.git
cd AlgoEval-GCD
```
2. Install Dependencies
   
`pip install -r requirements.txt`

4. Run Experiments

- Preprocess the cluster dataset:

`python google_cluster.py`

- Run baseline ML/AI models:

`python ai+cloud.py`

- Apply ensemble learning approaches:

`python ensemble_with_google.py`

- Borg-specific analysis:

`python googleborg.py`

Results

- Evaluation performed on Google Cluster Data v3.

- Algorithms compared: baseline ML models, cloud-specific AI methods, ensemble learning.

- Metrics: accuracy, F1, workload prediction efficiency.

( tables/figures once experiments )


Citation

If you use this repository in your research, please cite:
```
@misc{algoevalgcd2025,
  title   = {AlgoEval-GCD: Algorithm Evaluation on Google Cluster Data},
  author  = {Mohammed A.S. Al-Hitawi, Ahmed Hadi},
  year    = {2025},
  url     = {https://github.com/Mohammed20201991/AlgoEval-GCD}
}

```
