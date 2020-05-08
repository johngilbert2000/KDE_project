# KDE_project

This project improves the performance speed of Junjie's (俊傑) implementation of Elevated Relaxed Variable Kernel Density Estimation (E-RVKDE), Relaxed Variable Kernel Density Estimation (RVKDE), and Abramson's variable bandwidth Kernel Density Estimation (Abramson KDE) methods. The code in this repository was inspired heavily by the work of Junjie and Yen-Jen Oyang.

Elevated Relaxed Variable Kernel Density Estimation (E-RVKDE) is a KDE method developed by Junjie, under the advisor Yen-Jen Oyang at NTU. He demonstrated that, in most conditions, it produces results either on par with or better than Silverman's fixed bandwidth KDE (the default KDE of scipy). E-RVKDE was based upon earlier work by Yen-Jen Oyang et al. who originally developed the RVKDE method.

#### Example results for single gaussian, 1600 samples:
![KDE results](https://github.com/johngilbert2000/KDE_project/blob/master/kde_results/kde_plots/kde_1600_0.png)

#### Average Mean Square Error for Single Guassian

Samples |	Runs |	Total Time (sec) |	Silverman | RVKDE |	ERVKDE |	Abramson
--- | --- | --- | --- | --- | --- | ---
1600 | 10 | 105.064 | 1.265560E-05 | 2.830118E-05 | 1.143690E-05 | 2.762575E-05
3200 | 10 | 304.483 | 8.085018E-06 | 2.477353E-05 | 7.032016E-06 | 1.315627E-05
6400 | 10 | 1000.965 | 4.656962E-06 | 2.585347E-05 | 4.082943E-06 | 7.097434E-06
12800 | 10 | 3615.178 | 2.965795E-06 | 2.444878E-05 | 2.580114E-06 | 3.526277E-06


### Dependencies

The following versions were used in development
```
Python 3.8.1
numpy 1.18.1
numba 0.49.0
scipy 1.4.1
pandas 1.0.1
matplotlib 3.1.3
jupyter 1.0.0
```

### Setup
- Install [anaconda](https://docs.anaconda.com/anaconda/install/)
- Activate conda from the command line (`source anaconda3/bin/activate` or `. anaconda3/bin/activate`)
- Create and activate a conda environment ( `conda create kde_env`; `conda activate kde_env`)
- Install dependencies (`conda install numpy scipy pandas matplotlib numba jupyter`)
- Clone the repository
- Open jupyter notebooks (`jupyter notebook`)
- Run `Refactored_KDE.ipynb`
