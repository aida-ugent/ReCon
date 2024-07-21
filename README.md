# ReCon
This repository contains the supplementary material, source code, installation and use instructions for the method presented in the papers: "ReCon: Reducing congestion in job recommendation with optimal transport" and "Scalable Job Recommendation with Lower Congestion using Optimal Transport".


## Citation ##

If you have used ReCon in your research, please cite the following papers:

```bibtex
@inproceedings{10.1145/3604915.3608817,
author = {Mashayekhi, Yoosof and Kang, Bo and Lijffijt, Jefrey and De Bie, Tijl},
title = {ReCon: Reducing Congestion in Job Recommendation Using Optimal Transport},
year = {2023},
isbn = {9798400702419},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3604915.3608817},
doi = {10.1145/3604915.3608817},
abstract = {Recommender systems may suffer from congestion, meaning that there is an unequal distribution of the items in how often they are recommended. Some items may be recommended much more than others. Recommenders are increasingly used in domains where items have limited availability, such as the job market, where congestion is especially problematic: Recommending a vacancy—for which typically only one person will be hired—to a large number of job seekers may lead to frustration for job seekers, as they may be applying for jobs where they are not hired. This may also leave vacancies unfilled and result in job market inefficiency. We propose a novel approach to job recommendation called ReCon, accounting for the congestion problem. Our approach is to use an optimal transport component to ensure a more equal spread of vacancies over job seekers, combined with a job recommendation model in a multi-objective optimization problem. We evaluated our approach on two real-world job market datasets. The evaluation results show that ReCon has good performance on both congestion-related (e.g., Congestion) and desirability (e.g., NDCG) measures.},
booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems},
pages = {696–701},
numpages = {6},
keywords = {Job recommendation, Congestion-avoiding recommendation},
location = {Singapore, Singapore},
series = {RecSys '23}
}
```

and

```bibtex
@ARTICLE{10504265,
author={Mashayekhi, Yoosof and Kang, Bo and Lijffijt, Jefrey and De Bie, Tijl},
journal={IEEE Access}, 
title={Scalable Job Recommendation With Lower Congestion Using Optimal Transport}, 
year={2024},
volume={12},
number={},
pages={55491-55505},
keywords={Recommender systems;Optimization;Aggregates;Task analysis;Costs;Probabilistic logic;Indexes;Jobs listings;Performance evaluation;Job recommendation;optimal transport;congestion;exposure fairness;aggregate diversity},
doi={10.1109/ACCESS.2024.3390229}}
```

## Note ##
Some of the results of the baselines have been updated. However, they do not affect the conclusions of the paper. The updated results are included in the Journal_supplementary_material.pdf
