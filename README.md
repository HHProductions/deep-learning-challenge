# deep-learning-challenge



    Overview of the analysis: Explain the purpose of this analysis.
	
	This assignment applies machine learning  to help Alphabet Soup improve success rate of selecting  applicants whose project  will be successful.
	This is achieved by by reading in a dataset consisting of 34299 rows of  historical applicant data, categorising it, preprocessing it,  performing training and testing of the data and altering  various variables of the analysis.

    Results: Using bulleted lists and images to support your answers, address the following questions:

    Data Preprocessing
        What variable(s) are the target(s) for your model?
		The target  is the IS_SUCCESSFUL  column, in which 1 represents success and 0  no success.
        What variable(s) are the features for your model?
		All other data are features, with the exception of EIN and NAME, which  are dropped.
        Those two  exceptions are neither targets nor features.

    Compiling, Training, and Evaluating the Model
        How many neurons, layers, and activation functions did you select for your neural network model, and why?
		For initial analysis I utilised:
		- 2 hidden layers
		- 8 and 5 nodes in each of these layers
		- RELU activation for each hidden layer and SIGMOID  for the output layer
		- 50 epochs
		
	The choices above are useful for initial analysis. They allow for quick analysis and RELU  is a good starting  point for activation.	
	Monitoring  accuracy  change willl identify whether more epochs are necessary 
		
		
        Were you able to achieve the target model performance?
		I was only able to  achieve an accuracy of about 73.1%	This was below the target of 75%.
        What steps did you take in your attempts to increase model performance?
		Numerous attempts were made to  improve prediction accuracy.  These include:
		- Adding extra hidden layers (up to 4)
		- increasing the number of nodes, e.g. 30, and 15 for 2 hidden layers
		- Altering te activation type
		- reducing the features data set (e.g drop application_type)
		- reducing the categories for features like ASK_AMT. 
			- Used binning of data
				# Create the bins in which Data will be held
				bins = [0, 5001, 50000, 100000, 500000, 1000000,20000000, 10000000000]
				# Create the names for the five bins
				group_names = ["5k", "50k", "100k", "500k", "1m","20m","over 20m"]
			-  dropped higher amounts (ew2_application_df = new_application_df.drop(new_application_df[new_application_df['ASK_AMT'] > 20000000].index))
			new2_application_df["binned_ASK_AMT"]= pd.cut(new2_application_df['ASK_AMT'], bins, labels=group_names, include_lowest=True)
new2_application_df
		
		I attempted various permutations and combinations and I  was not  able to improvide the accuracy  much above 73%. in must  instances it  remained within a range of 72.7 and 73.3%.

    Summary: Summarise the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
I was recommended to include NAME as a feature, however with over 19,000 names my computer did not have enough memory when creating x_train and y_train. (MemoryError: Unable to allocate 3.76 GiB for an array with shape (25724, 19617) and data type int64)
