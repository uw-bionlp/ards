# A PyTorch Wrapper of Hierarchical Attention Network with Sentence Objectives Framework for ARDS Identification
This repository implements a high-level wrapper using Python3 and PyTorch of methods described in the paper [Identifying ARDS using the Hierarchical Attention Network with Sentence Objectives Framework](https://arxiv.org/abs/2103.06352). 

The project was developed by [Professor Meliha Yetisgen and Dr. Kevin Lybarger of UW-BioNLP](http://depts.washington.edu/bionlp/index.html?people) and collaborators at the [University of Washington](http://www.washington.edu/).

# Requirements
 - Python version >= 3.6
 - PyTorch version >= 1.6.0
 - Transformers >= 4.0.1

 See [requirements.txt](requirements.txt) for additional dependencies.

# Installation
This repository provides a simple high-level Python-based wrapper command-line for processing clinical notes (as `.txt` documents) and outputting `.json` files of the predicted values.

1. As the trained model for this code is not included in the repository, please contact contact Professor Meliha Yetisgen [melihay@uw.edu](mailto:melihay@uw.edu) to gain access to the model.

2. Install Python3 dependencies. We recommend doing so using a virtual environment:

```sh
$ cd /ards
$ python3 -m venv venv
$ source venv/bin/activate
$ python3 -m pip install -r requirements
```

3. Download the spaCy english language model

```sh
$ python3 -m spacy download en
```

3. Run the tool on a directory of clinical notes as `.txt` files:

```sh
$ python3 main.py <your_notes_directory>
```

You can also specify the additional parameters
- `--output_path`, which outputs your files to a specify location (rather than a default `/output/<dir_name><current_timestamp> directory)
- `--output_single_file`, which can alternatively output a single `.jsonl` file with each row in the file representing an input file's output
- `--batch_size`, which specifies the number of note batches to predict at a time. If unspecified, this defaults to `8`.