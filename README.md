# HGTMAE
<p align="center">
  <img src="figures/front_figures/high_res_plot_DISCO_KODE.png" alt="DISCO KODE Plot" width="45%" style="margin-right: 10px;">
  <img src="figures/front_figures/high_res_plot_Q_BREDT_LOEN_BELOEB.png" alt="Q BREDT LOEN BELOEB Plot" width="45%">
</p>

This repository provides an implementation of the Heterogeneous Graph Transformer Masked AutoEncoder (HGTMAE) model as described in our master's thesis, *"Large-Scale Complex Network Embeddings: Leveraging Graph Representation Learning To Predict Future Life Events."* 

## General Information

Please note that this repository contains a simplified version of our model. The full implementation used in our research at Statistics Denmark is not included due to privacy concerns. As a result, some components, such as data preprocessing, have been omitted. Instead, this repository offers a basic implementation of HGTMAE, alongside a toy IMDB graph dataset, to demonstrate how the model can be trained.

**Important Notes:**

- The embeddings generated from our thesis are available as high resolution images in the `figures` directory.
- The toy version provided here is for demonstration purposes only and should not be regarded as a rigorous implementation. The results generated from this repository are intended to provide an impression of how the model can be trained, rather than accurate or high-performance embeddings.

## Prerequisites

To train the model, you need to create a Conda environment using the provided YAML file. This environment contains all the necessary dependencies.

### Steps to Set Up the Environment:

1. **Create Conda Environment:**
   ```bash
   conda env create -f hgn.yml
   conda activate hgn
   ```

## Training the Model

You have two options for training the model:

### Option 1: Training from Scratch

This approach involves running the entire pipeline from scratch, starting with generating batches from a DGL graph object.

#### Steps:

1. **Generate Batches:**
   Run the sample script to generate the necessary batches from the provided graph.
   ```bash
   python src/sample.py --graph_path "toy_graphs/imdb.bin"
   ```
   This script will create a dictionary under the `toy_graphs/batches/time_stamp/` directory.

2. **Train the Model:**
   After generating the batches, you can train the model using the following command:
   ```bash
   python src/main.py --batch_path "toy_graphs/batches/time_stamp/"
   ```
   This will train the HGTMAE model on the toy dataset.

3. **Inspect the Learned Embeddings:**
   You can explore the learned embeddings using the provided Jupyter notebook:
   ```bash
   jupyter notebook notebooks/imdb_analysis.ipynb
   ```
   Remember to update the model paths to your saved checkpoints, especially if you trained the model from scratch. Also, adjust the model configurations in the notebook if you deviated from the default settings.

### Option 2: Training with Provided Batches

This option allows you to train the model using pre-generated batches included in the repository.

#### Steps:

1. **Train the Model:**
   Simply run the main script with the provided batches:
   ```bash
   python src/main.py
   ```
   This will use the pre-generated batches to train the model.

2. **Inspect the Learned Embeddings:**
   As with the first option, you can inspect the embeddings using the `imdb_analysis.ipynb` notebook. Again, make sure to update the checkpoint paths if you want to analyze the results from your recent training session.

## Additional Notes

- **Data Format:** Ensure that any graph you use is in the DGL format with available node features. This is essential for the correct functioning of the scripts provided.
- **Checkpoints and Configurations:** Always verify that paths to model checkpoints and configurations match your training setup, especially if you are training from scratch.

---

This guide should help you get started with using the HGTMAE repository.