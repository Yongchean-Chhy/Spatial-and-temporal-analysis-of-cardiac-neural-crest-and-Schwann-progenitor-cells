# Spatial-and-temporal-analysis-of-cardiac-neural-crest-and-Schwann-progenitor-cells
Reqirements:
- anndata
- matplotlib
- mygene
- numpy
- pandas
- scanpy
- scikit-image
- scikit-learn
- scipy
- seaborn
- squidpy
- tifffile
- umap-learn

Datasets:
• Spatial Transcriptomics Dataset (Version3): https://data.mendeley.com/datasets/compare/mbvhhf8m62?old=2&new=3
• In Situ Sequencing (ISS) Dataset: https://figshare.com/articles/dataset/ISS_data_in_A_spatiotemporal_organ-wide_gene_expression_and_cell_atlas_of_the_developing_human_heart_/10058048/1

├── README.md
├── code
├── plots
├── results
├── dataset
    ├── dataset_iss
        ├── dapi_data
        ├── spots_data
    ├── dataset_st
        Filtered
            ├── Developmental_heart_filtered_scRNA-seq_and_meta_data
            └── filtered_ST_matrix_and_meta_data

Downloads the required libraries. Download the datasets and put in the correct directory.
To run ST pipeline: python3 st.py
To run ISS pipeline: python3 iss.py
To run cross modality validation: python3 cross_modality_analysis.py
To run evaluation: python3 evaluation.py
