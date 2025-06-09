This testing pipeline is part of the masters dissertation: "Exploring Grammatical Constraints with Large Language Models" by David Silva, supervised by João Veríssimo and Vânia Mendonça.

It tests for the Plurals in Compounds effect and uses the surprisal package: 
https://github.com/aalok-sathe/surprisal/tree/main

Install all requested packages using pip install.

Run 'test_pic_model.py' files to obtain the results for each model.

Run 'mean_SD_suprisal_model.py' files to obtain the mean surprisal values and standard deviation.
Run 'mean_SD_suprisal_model_dif.py' files to obtain the same statistics for the diference between plurals and singulars for regular and irregular conditions. 

The lists of heads or non_heads to be tested can be seen or changed in the respective csv files: list_type.csv
