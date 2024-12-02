# MLGIG VM 2024 data challenge submission
This repo provides code, slides and results for our winning entry in the International Data Challenge on Spectroscopy and Chemometrics 2024.

MLGIG Team – Milk Lactose Prediction Data Challenge, 4th International Workshop on Spectroscopy and Chemometrics 2024, organised by the VistaMilk SFI Research Centre.

The methods and results are described in our paper [Lactose prediction in dry milk with hyperspectral imaging: A data analysis competition at the “International Workshop on Spectroscopy and Chemometrics 2024”](https://www.sciencedirect.com/science/article/pii/S0169743924002193#d1e1642).

The data can be downloaded [here](https://drive.google.com/drive/folders/10Bv-trk4wb9GjK1GjUaXRvDsSU9LxyZX?usp=sharing)

## Citation
If you use this work, please cite as:
```
@article{FRIZZARIN2024105279,
title = {Lactose prediction in dry milk with hyperspectral imaging: A data analysis competition at the “International Workshop on Spectroscopy and Chemometrics 2024”},
journal = {Chemometrics and Intelligent Laboratory Systems},
pages = {105279},
year = {2024},
issn = {0169-7439},
doi = {https://doi.org/10.1016/j.chemolab.2024.105279},
url = {https://www.sciencedirect.com/science/article/pii/S0169743924002193},
author = {Maria Frizzarin and Vicky Caponigro and Katarina Domijan and Arnaud Molle and Timilehin Aderinola and Thach Le Nguyen and Davide Serramazza and Georgiana Ifrim and Agnieszka Konkolewska},
keywords = {Infrared spectroscopy, Hyperspectral imaging, Lactose concentration, Machine learning},
abstract = {In April 2024, the Vistamilk SFI Research Centre organized the fourth edition of the “International Workshop on Spectroscopy and Chemometrics – Spectroscopy meets modern Statistics”. Within this event, a data challenge was organized among workshop participants, focusing on hyperspectral imaging (HSI) of milk samples. Milk is a complex emulsion comprising of fats, water, proteins, and carbohydrates. Due to the widespread prevalence of lactose intolerance, precise lactose quantification in milk samples became necessary for the dairy industry. The dataset provided to the participants contained spectral data extracted from HSI, without the spatial information, obtained from 72 samples with reference laboratory values for lactose concentration [mg/mL]. The winning strategy was built using ROCKET, a convolutional-based method that was originally designed for time series classification, which achieved a Pearson correlation of 0.86 and RMSE of 9.8 on the test set. The present paper describes the approaches and statistical methods adopted by all the participants to analyse the data and develop the lactose prediction models.}
}


The dataset used in this data challenge was collected and described in the paper below. 
Please cite as:

@article{CAPONIGRO2023109351,
title = {Single-drop technique for lactose prediction in dry milk on metallic surfaces: Comparison of Raman, FT – NIR, and FT – MIR spectral imaging},
journal = {Food Control},
volume = {144},
pages = {109351},
year = {2023},
issn = {0956-7135},
doi = {https://doi.org/10.1016/j.foodcont.2022.109351},
url = {https://www.sciencedirect.com/science/article/pii/S0956713522005448},
author = {Vicky Caponigro and Federico Marini and Amalia G.M. Scannell and Aoife A. Gowen},
keywords = {Milk, Lactose, Raman, FT, NIR, FT-MIR, Spectral, Hyperspectral, Imaging, Aluminium, Stainless steel, PLS},
abstract = {This study applies the single drop techniques to compare the efficacy of Raman, FT – NIR, and FT-MIR spectral imaging to quantify lactose concentration in dried whole milk on different metallic surfaces. Drying the samples avoids degradation problems such as water evaporation or oil degradation and scattering due to micelles. Spectral imaging techniques minimise sampling issues while also describing the sample spatial variation. The mean spectra of pre-processed images were used to build PLS regression models to predict lactose concentration. Raman, FT – NIR (5600–3730 cm−1), FT–MIR (3533–600 cm−1) models and the model obtained using the fusion of the three ranges were built independently and compared. This study confirms that is possible to quantify lactose rapidly using spectral imaging without adding standard references: the minimum RMSEP = 2.8 mg/mL (R2 = 0.98) was achieved with FT – MIR spectral imaging.}
}


```
