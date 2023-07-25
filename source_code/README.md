# ReCon
This repository contains the source code, installation and use instructions for the method presented in the paper: "ReCon: Reducing congestion in job recommendation with optimal transport". We provide Python implementations of the complete ReCon model.


# Source code for ReCon
Python 3.10

From the project root:

To create the environment

    - conda env create --name venv_name --file  requirements.txt

Then run:

    - Conda activate venv_name
    - mkidr models
    - mkdir figures

To run CNE without optimal transport loss:

    - python recommendation_method/cne/main.py --use_ot 0

To run CNE with optimal transport loss:

    - python recommendation_method/cne/main.py --use_ot 1
