# What does it doing? 

We describe a data-driven discovery method that leverages Simpson's paradox to uncover interesting patterns in behavioral data. Our method systematically disaggregates data to identify subgroups within a population whose behavior deviates significantly from the rest of the population. 
Given an outcome of interest and a set of covariates, the method follows three steps. First, it disaggregates data into subgroups, by conditioning on a particular covariate, so as minimize the variation of the outcome within the subgroups. Next, it models the outcome as a linear function of another covariate, both in the subgroups and in the aggregate data. Finally, it compares trends to identify disaggregations that produce subgroups with different behaviors from the aggregate.

More details about the algorithm is available in our paper on ICWSM 2018, "Using Simpson’s Paradox to Discover Interesting Patterns in Behavioral Data".


# Input and Running: 
For running the algorithm you need to do the following:
### Step 1: 
Add .csv data file to input/ directory

	- Put your data file (.csv) into input/ directory
	- Your data file must consists of the header row for name of the variables, each row is a datapoint.
	- .csv datafile (with first row column names) 
	- Optional binning parameters 
	  - Number of bins (nbins)
	  - Minimum number of datapoint per bin (mindatapoints)
	  - Lambda parameter (smaller lambda => finer binning (lambda)

### Step 2: 
Update input_info.json file

	- num_of_bins: Maximum number of bins for disaggregating data
	- least_num_of_datapoints_in_each_bin: Minimum number of datapoints in each bin
	- target_variable: name of the target variable in input .csv file
	- target_variable_column: Column number of target variable in .csv file (zero-based).
	- level_of_significance: Level of significance for chi-square deviance test
	- csv_file_name: Name of the file you put to the input/ directory in step 1
	- ignore_columns: An array of name of the variables for not including them in the algorithm variables. You should list all the columns with string or float values.
	- log_scales: A dictionary of variable name to boolean value. Which shows you prefer log scale for axis for that variable in output/ plots or not. 

	Before running the algorithm, please make sure the json format of the input_info.json file is valid using: https://jsonformatter.curiousconcept.com/

### Step 3: 
Run the run.sh script 

# Output: 

Plots of Trend Simpson's Pairs will be available in output/ directory. For each pair, there is a PDF file. First plot is logistic fit to aggregated data. Second one is logistic for for each of the bins. The third plots are histogram and heatplot of the Paradox and Conditioning variables. 
The details of the algorithm will be printed in the terminal. You can use them as log. Beside that, all the informations about the logistic fits, simpson's pairs and deviance values are availabe in store_results/ directory as python pickle objects. You can use load function in trend_simpsons.py file for loading them. 
