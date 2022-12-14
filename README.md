# scDEFR
# Unsupervised Deep Embedded Fusion Representation of Single-cell Transcriptomics
Cell clustering is a critical step in analyzing single-cell RNA sequencing (scRNA-seq) data that allows characterization of the cellular heterogeneity after transcriptional profiling at the single-cell level. Single-cell deep embedded representation models have gained popularity recently as they can learn feature representation and clustering simultaneously. However, the models still pose a variety of significant challenges, including the massive amount of data, pervasive dropout events, and complicated noise patterns in transcriptional profiling. Here, we propose a Single-Cell Deep Embedding Fusion Representation (scDEFR) model that produces a deep embedded fusion representation to learn the fused heterogeneous latent embedding containing both the gene-level transcriptome and cell topology information. We first fuse them layer by layer to obtain compressed representations of the intercellular relationships and transcriptome information. Then, we use the zero-inflated negative binomial model (ZINB)-based decoder to capture the global probabilistic structure of the data to reconstruct the final gene expression information. Finally, by simultaneously integrating the clustering loss, cross-entropy loss, ZINB loss, and cell graph reconstruction loss, scDEFR can optimize clustering performance and learn the latent representation from the fused information in a joint mutual supervised strategy. We conducted comprehensive experiments on 15 single-cell RNA-seq datasets from different sequencing platforms and demonstrated the superiority of scDEFR over a variety of state-of-the-art methods.
## Installation
### pip
        pip install -r requirements
### usage
##### You can run the scDEFR from the command line:
        python sigmatrain.py --dataname Qs_Limb_Muscle
### reproduct results
##### You can find the pretrained model parameters of the Qs_Limb_Muscle in the release.
