# StockMarketGNN

This initiative represents a research endeavor undertaken as part of a thesis at the University of São Paulo, with the objective of examining the potential of Graph Neural Networks (GNNs) in forecasting stock market prices. The research specifically focuses on the implementation of Graph Attention Networks (GATs) across various clusters of interconnected Brazilian stocks. This comprehensive project encompasses the whole pipeline of the analytical process, ranging from the initial data ingestion into an SQLite database to the execution of predictive modeling for subsequent time intervals.

## Setup

This section will guide you through the initial setup of the project.

### Prerequisites

- Git
- Docker

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/valentincthrn/StockMarketGNN.git
```

2. **Start the application using Docker**

```bash
docker-compose up --build
```

## Usage

This project can be used either through a Streamlit interface or via command line.

### Through Streamlit Interface

- Navigate to `http://localhost:[8501]/` to access the Streamlit interface.

### Through Command Line

#### `stock_predictions` Function Explanation

The `stock_predictions` function is the main entry point for running the pipeline, from data ingestion to model training for stock predictions. This function is designed to be flexible and configurable through various command line arguments.

#### Usage

To run the `stock_predictions` function, use the following command:

```bash
python run.py stock-predictions [OPTIONS]
```

#### Options

- `-c`, `--config-path` (default: `params/run_config.yml`): Path to the run configuration file. This file contains all the necessary configurations for the run.
  
  ```bash
  --config-path 'path/to/config.yml'
  ```

- `-i`, `--ignore-ingest` (default: `False`): Flag to determine whether to ignore the data ingestion step. Use this option if you already have the required data ingested.

  ```bash
  --ignore-ingest
  ```

- `-s`, `--stocks-group` (choices: `Banks`, `Distinct`, `FromConfig`; default: `FromConfig`): Selects the group of stocks to use for the predictions. If set to `FromConfig`, the stocks are taken from the run configuration file.

  ```bash
  --stocks-group 'Banks'
  ```

- `-m`, `--macro` (choices: `All`, `FromConfig`, `Not`; default: `FromConfig`): Determines which macroeconomic indicators to include. Use `All` to include all indicators, `FromConfig` to use settings from the config file, or `Not` to exclude macroeconomic indicators.

  ```bash
  --macro 'All'
  ```

- `-f`, `--fund` (choices: `All`, `FromConfig`, `Not`; default: `FromConfig`): Specifies which fundamental indicators to include. Similar to the `macro` option, this can be set to `All`, `FromConfig`, or `Not`.

  ```bash
  --fund 'All'
  ```

- `-e`, `--exp-name` (default: `Test`): Name of the experiment. This name is used for logging and tracking the experiment runs.

  ```bash
  --exp-name 'Experiment1'
  ```

- `--debug`/`--no-debug` (default: `False`): Enables or disables debug logging. Useful for development or troubleshooting.

  ```bash
  --debug
  ```

- `--force` (default: `False`): Forces the regeneration of the SQL database. Use this if you need to refresh your database with new data.

  ```bash
  --force
  ```

#### Function Description

The `stock_predictions` function configures and initiates the entire pipeline, including data ingestion, preprocessing, and model training. It handles different configurations for stocks, macroeconomic and fundamental indicators based on the provided arguments. The function ensures flexibility and customizability for different experimental setups and data configurations. 

##### Important Notes

- The function utilizes PyTorch, and will automatically use CUDA if available.
- Target stocks, macro, and fundamental indicators can be dynamically configured via the command line or the configuration file.
- The function provides informative logging, especially useful when `--debug` is enabled.
- Ensure that the `params/run_config.yml` file is correctly set up according to your experiment requirements.


## Project Explanation

The primary objective of this project is to conduct an exploratory analysis of the application of Graph Neural Networks (GNNs) in stock prediction. Recent endeavors in this domain suggest that GNNs may possess the capability to encapsulate the intricate dynamics of stock market movements. Consequently, this project is designed to investigate the following critical inquiries:

- Do GNNs exhibit enhanced learning capabilities when the input graph is constructed from stocks within the same sector?
- To what extent do fundamental and macroeconomic indicators contribute to the improvement of predictive outcomes in this context of GNNs?
  
By addressing these questions, the project aims to shed light on the efficacy of GNNs in financial modeling and provide insights into the significance of sector-specific correlations and external economic factors in stock prediction algorithms.

### Model

This section presents a detailed overview of the predictive model architecture. 

![Image](https://github.com/valentincthrn/StockMarketGNN/blob/main/images/model.png)

The model is intricately designed and comprises three primary modules:
  1. Feature Extraction
  2. Graph Attention Network (GAT)
  3. Prediction Head
     
#### Module 1: Feature Extraction
This initial module is dedicated to constructing a robust embedding for each corporation by leveraging its distinctive features. Essentially, it transforms a suite of attributes for each entity into a cohesive, fixed-size, one-dimensional embedding through a Gated Recurrent Unit (GRU), which subsequently feeds into the GAT module.

Every corporation's dataset incorporates its daily stock price movements. Notably, for preselected banking groups, additional historical fundamental data has been meticulously curated from the StatusInvest platform.

#### Module 2: GAT
Here, an interconnected graph is established, positioning the stocks as nodes with the one-dimensional vectors from the first module as their features. Utilizing the GAT mechanism provided by PyTorch, the model engenders refined node embeddings based on the graph's structure and node features.

#### Module 3: Prediction Head
The final module of the architecture harnesses the node embeddings generated by the GAT. It processes each embedding through a Multilayer Perceptron (MLP) to project the subsequent 'K' time steps, with 'K' being defined within the run_config.yml configuration file.

### Results

WIP

### Contact

This project constitutes a component of a broader research initiative. While my direct involvement may conclude with the research phase, I welcome any inquiries or clarifications regarding the project. Should you have any questions or require further information, please do not hesitate to reach out to me at valentincthrn@gmail.com.

---
