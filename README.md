# Scientific Writing: Relative Entropy Analysis

Relative entropy analysis on the royal society corpus using python/scipy/matplotlib

Dependencies:

- Python3
- bs4
- matplotlib
- scipy

Required Files:

- Royal society corpus metadata tsv available for download [here](https://fedora.clarin-d.uni-saarland.de/rsc_v6/access.html#download)
- Royal society corpus tei xml files for each text, also available for download [here](https://fedora.clarin-d.uni-saarland.de/rsc_v6/access.html#download)

Setup:

- Make sure ROOT_DIR is set to the location of the metadata file
- Make sure all tei xml files are within a single folder that is a direct child of the root directory, and TEXT_DIR is set to the name of that directory
- Make sure the METADATA_FILENAME and TEXT_BASE_FILENAME parameters match the name of the downloaded files - If column titles change, set the column names of the ID_COL, ..., DECADE_COL
- Tune parameters if desired, defaults are recommended values
- Set output filenames to desired values
