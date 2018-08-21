#!/bin/bash


scp -r config_cnn.yaml cnn_train.py constants.py  data_augumentation.py data_loading.py data_cleaning.py db_functions.py plot.py simple_models.py utils.py rep_counting.py soroa@tik42x.ethz.ch:/usr/itetnas03/data-tik-01/soroa/

# for file_name in ${files_to_upload[@]}:
# do
	# scp -r file_name  soroa@tik42x.ethz.ch:/usr/itetnas03/data-tik-01/soroa/
	# echo $file_name
# done	
