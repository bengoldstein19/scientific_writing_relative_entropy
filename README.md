# Scientific Writing: Relative Entropy Analysis

Final paper is available for download in the "Goldstein Benjamin LING 219 Final Paper.docx" file (I previously submitted as a word doc to run turnitin but figured better to have everything in one place). All plots are in the plots folder, all output data csvs are in the output_data folder. The repo can be downloaded as a zip or files can be downloaded individually.

Relative entropy analysis on the royal society corpus using python/scipy/matplotlib

Dependencies:

- Python3.x
- bs4
- matplotlib
- scipy

Required Files:

- Royal society corpus metadata tsv available for download [here](https://fedora.clarin-d.uni-saarland.de/rsc_v6/access.html#download)
- Royal society corpus tei xml files for each text, also available for download [here](https://fedora.clarin-d.uni-saarland.de/rsc_v6/access.html#download)

Setup:

- Clone repository by running git clone https://github.com/bengoldstein19/scientific_writing_relative_entropy.git
- [Install Python3.x](https://www.python.org/downloads)
- Install dependencies by running pip install -r requirements.txt
- Make sure ROOT_DIR is set to the location of the metadata file
- Make sure all tei xml files are within a single folder that is a direct child of the root directory, and TEXT_DIR is set to the name of that directory
- Make sure the METADATA_FILENAME and TEXT_BASE_FILENAME parameters match the name of the downloaded files - If column titles change, set the column names of the ID_COL, ..., DECADE_COL
- Tune parameters if desired, defaults are recommended values
- Set output filenames to desired values
- Run from the command line using either ./analysis.py (make sure you've given analysis.py executable permissions) or python analysis.py
