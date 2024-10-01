# ReCon
This repository contains the source code, installation and use instructions for the method presented in the papers: "ReCon: Reducing congestion in job recommendation with optimal transport" and "Scalable Job Recommendation with Lower Congestion using Optimal Transport". We provide Python implementations of the complete ReCon model.


# Source code for ReCon
Python 3.10

From the source_code folder:

To create the environment

    - pip install -r requirements.txt

Then run:

    - mkdir models

To run CNE without optimal transport loss:

    - python recommendation_method/cne/main.py --use_ot 0

To run CNE with optimal transport loss computed for all users and items:

    - python recommendation_method/cne/main.py --use_ot 1 --ot_method all
    
To run CNE with optimal transport loss computed only for users and items involved in each batch:

    - python recommendation_method/cne/main.py --use_ot 1 --ot_method batches


## Note ##
Please note that the small datasets used for the experiments are subsets of the datasets explained in the papers.

