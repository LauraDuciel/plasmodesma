# Introduction
###  Machine Learning in dereplication and complex mixture analysis.
Dereplication - or counterscreening - is a major field in natural product discovery with applications mainly in Mass Spectrometry and Nuclear Magnetic Resonance.
The aim of the technique is to accelerate the drug discovery process [Gaudêncio et al. 2015]. 
In MS, molecular networking is one of the technique developed to reach this goal. 
It compares MS/MS spectrum of natural product mixture to known MS/MS spectra standards. 
This thus allows to predict which molecules are present in the considered mixture [Yang et al., 2013]. 
Machine learning approaches are used to perform that prediction. 
These solutions can be extended to the elaboration of spectral networks [Mohimania et al. 2016].
Similar approaches are indeed developed in the NMR field to perform early detection of known compounds in complex plant extracts [Bakiri et al. 2017] [Wolfender et al. 2018]. 
Our development anchors in this approach as we are using machine learning application to obtain the molecular footprint of active molecule from unrefined fractions of the sample.

### Hyperparameters optimization.
Automatic large scale analysis of big datasets requires the development of machine learning algorithms which are based on models and multiple parameters. 
These parameters are of great importance and can highly impact the results and performances of the program. 
These parameters are called hyper-parameters of the algorithm and should be refined, optimized to get the better results in the given application [Luo 2016]. 
Indeed, as in any scientific research, the chosen model is as important as the dataset on which the analysis is applied. To perform this optimization, some iterations have to be done manually and the results must be interpreted and evaluated by experts in the domain of application. 
In some general cases, algorithms have been developed to help non-expert either in computing or in the field choosing an optimized set of parameters [Bergstra et al. 2011]. 
For example, it is well developed for Neural Network learning which is a machine learning technique wide-spread in the domain [Domhan et al. 2015].

# Material & Methods
### Data Processing & Analysis
The spectral processing presented in this article is realised mainly with the **Plasmodesma** Program [Margueritte et al., 2017], written in **Python 3.6.5 | Anaconda, Inc. | Apr 29 2018**. 
It is a specific programming language conceived to be powerful, versatile, but also easy to learn and apply, offering a lot of programming possibilities [Langtangen, 2009]. 
Plasmodesma was based on the **SPIKE Processing Library | Version 0.99.0. | Date 06.03.2018 | Revision Id 369**. 
SPIKE, standing for "Spectrometry Processing Innovative KErnel" [Chiron et al., 2016], increments tools to process and visualize NMR and Mass Spectrometry datasets from raw data.
Plasmodesma produces peak lists and bucket lists which are analysed mainly with **Scikit-Learn | 0.20.1**, a specific python library providing tools for data analysis and application of machine learning techniques [Pedregosa et al. 2011]. 
Our implementation provides a linear regression (sklearn.linear_model.LinearRegression) over the complete set of data provided as well as a Recursive Feature Elimination (sklearn.feature_selection.RFE). 
**! TO BE COMPLETED - RFE without negatives, Logistic Regression ? !**

### Data Visualization
In this paper we present an enrichment of the processing part with a Graphical User Interface to use the Plasmodesma Program.
This interface is obtained using **Bokeh | 1.0.1**, a python library for interactive visualization on web devices, and can be accessed at http://**TOBECOMPLETED**.temporary.fr/. 
It allows to get elegant figures from Python data that can be combined into a complete Bokeh application embeddable in an online accessible web server while keeping an interactivity. 
As an example, it was previously used to create nice and fully interactive display of spectrum and van Krevelen diagrams for mass spectrometry data available at *https://wkew.github.io/FTMSViz/SRFA-plot.html* [Kew et al., 2017]. 

# Bibliography
--------
### Intro part
##### Machine Learning in dereplication and complex mixture analysis.
**[Gaudêncio et al. 2015]** Gaudêncio, S.-P. and Pereira, F. (2015). Dereplication: racing to speed up the natural products discovery process. Nat. Prod. Rep. 32 (6), 779-810.
**[Yang et al., 2013]** Yang, J.-Y., Sanchez, L.-M., Rath, C.-M., Liu, X., Boudreau, P.-D., Bruns, N., Glukhov, E., Wodtke, A., de Felicio, R., Fenner, A., Ruh Wong, W., Linington, R.-G., Zhang, L., Debonsi, H.-M., Gerwick, W.-H. and Dorrestein, P.-C. (2013). Molecular Networking as a Dereplication Strategy. J. Nat. Prod. 76 (9), 1686–1699.
**[Bakiri et al. 2017]** Bakiri, A., Plainchont, B., de Paulo Emerenciano, V., Reynaud, R., Hubert, J., Renault, J.-H., Nuzillard, J.-M. (2017). Computer‐aided Dereplication and Structure Elucidation of Natural Products at the University of Reims. Mol. Inf. 36, 1700027.
**[Wolfender et al. 2018]** Wolfender, J.-L., Nuzillard, J.-M., van der Hooft, J.-J.-J., Renault, J.-H. and Bertrand, S. (2018). Accelerating metabolite identification in natural product research: toward an ideal combination of LC-HRMS/MS and NMR profiling, in silico databases and chemometrics. Anal. Chem. **Just Accepted Manuscript**
**[Mohimania et al. 2016]** Mohimania, H. and Pevzner, P.-A. (2016). Dereplication, sequencing and identification of peptidic natural products: from genome mining to peptidogenomics to spectral networks. Nat. Prod. Rep. 33, 73-86. 
##### Hyperparameters optimization.
**[Luo 2016]** Luo, G. (2016). A review of automatic selection methods for machine learning algorithms and hyper-parameter values. Netw Model Anal Health Inform Bioinforma 5: 18.
**[Bergstra et al. 2011]** Bergstra, J., Bardenet, R., Bengio, Y. and Kégl, B. (2011). Algorithms for hyper-parameter optimization. In Proceedings of the 24th International Conference on Neural Information Processing Systems (NIPS'11), J. Shawe-Taylor, R. S. Zemel, P. L. Bartlett, F. Pereira, and K. Q. Weinberger (Eds.). Curran Associates Inc., USA, 2546-2554.
**[Domhan et al. 2015]** Domhan, T., Springenberg, J.-T. and Hutter, F. (2015). Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves. In Proceedings of the 24th International Conference on Artificial Intelligence (IJCAI'15), Qiang Yang and Michael Wooldridge (Eds.). AAAI Press 3460-3468.
- https://scikit-optimize.github.io/notebooks/hyperparameter-optimization.html
--------
### Mat&Met part
**[Langtangen, 2009]** Langtangen, H.-P. (2009). A primer on scientific programming with Python, vol. 2,.Springer
**[Margueritte et al., 2017]** Margueritte, L., Markov, P., Chiron, L., Starck, J.-P., Vonthron-Sénécheau, C., Bourjot, M., Delsuc, M.-A. (2017). Automatic differential analysis of NMR experiments in complex samples. Magn. Reson. Chem. 1-11.
**[Chiron et al., 2016]** Chiron, L., Coutouly, M.-A., Starck, J.-P., Rolando, C. and Delsuc, M.-A. (2016). SPIKE a Processing Software dedicated to Fourier Spectroscopies. preprint arXiv:1608.06777 Phys. .
**[Kew et al., 2017]** Kew, W., Blackburn, J.-W.-T., Clarke, D.-J. and Uhrín, D. (2017). Interactive van Krevelen diagrams–Advanced visualisation of mass spectrometry data of complex mixtures. Rapid. Commun. Mass Spectrom. 31, 658–662.
**[Pedregosa et al. 2011]** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 2825-2830.
-------
