# Polymer Solar Cells Machine learning

The format of the data sets released as part of this work is described below. The data is extracted and curated from the abstracts of journal papers.

## Description of dataset columns

- **DOI**: Unique digital identifier for each paper that the datapoint is taken from. Each paper can be accessed by the hyperlink https://doi.org/{DOI}

- **curated**: If 1, then the corresponding document has been curated by a domain expert

- **skip**: If 1, then the document was examined by a domain expert and found to lie outside the scope of this study

- **donor**: Name of donor

- **donor_smiles**: SMILES string of donor in the psmiles notation as described in Ref. [[1]](https://www.nature.com/articles/s41467-023-39868-6)

- **donor_coreferents**: All names by which the same donor is referred to in that particular abstract

- **fullerene_acceptor**: If 1, then the acceptor reported in that row is a fullerene acceptor.

- **acceptor**: Name of acceptor

- **acceptor_smiles**: SMILES string of acceptor. SMILES strings for fullerene acceptors are not provided here as they are very large and instead are provided in datasets/constants.py

- **acceptor_coreferents**: All names by which the same acceptor is referred to in that particular abstract

- **property_metadata**:  A json string with metadata about the location in the abstract from which property values were extracted by the NLP pipeline

## References

[1] Kuenneth, C., & Ramprasad, R. (2023). polyBERT: a chemical language model to enable fully machine-driven ultrafast polymer informatics. Nature Communications, 14(1), 4099.
