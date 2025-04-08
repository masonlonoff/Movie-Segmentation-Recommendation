# DSGA1004 - BIG DATA
## Capstone project

*Handout date*: 2025-04-07

*Checkpoint submission*: 2025-04-28

*Project due date*: 2025-05-07


# Overview

In the capstone project, you will apply the tools you have learned in this class to solve a realistic, large-scale applied problem.
Specifically, you will use the movielens dataset to build and evaluate a collaborative-filter based recommender as well as a customer segmentation system. 

In either case, you are encouraged to work in **groups of up to 4 students**:


## The data set

In this project, we'll use the [MovieLens](https://grouplens.org/datasets/movielens/latest/) dataset provided by F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

We have prepared two versions of this dataset for you in Dataproc's HDFS: 
A small dataset for prototyping is at /user/pw44_nyu_edu/ml-latest-small.zip (9000 movies, 600 users)
The full dataset for scaling up is at /user/pw44_nyu_edu/ml-latest.zip (86000 movies and 330000 users)

Each version of the data contains rating and tag interactions, and the larger sample includes "tag genome" data for each movie, which you may consider as additional features beyond
the collaborative filter. Each version of the data includes a README.txt file which explains the contents and structure of the data which are stored in CSV files.
We strongly recommend to thoroughly read through the dataset documentation before beginning, and make note of the documented differences between the smaller and larger datasets.
Knowing these differences in advance will save you many headaches when it comes time to scale up.
Note: In general, use the small dataset for prototyping, but answer the questions below by using the full dataset.
Also note that these files are provided to you as zip files for ease - they both unzip as larger folders with many files. You should copy these files to your local hdfs and unzip them there by commands like these:

hadoop fs -copyToLocal /user/pw44_nyu_edu/ml-latest.zip .

unzip ml-latest.zip

hadoop fs -copyFromLocal ./ml-latest /user/[YOUR_NETID]_nyu_edu/target


## What we would like you to build / do (all 5 deliverables are equally weighed)

## Customer segmentation

1.  Customer segmentation relies on similarity, so we first want you to find the top 100 pairs of users ("movie twins") who have the most similar movie watching style. Note: For the sake of simplicity, you can operationalize "movie watching style" simply by the set of movies that was rated, regardless of the actual numerical ratings. We strongly recommend to do this with a minHash-based algorithm.
2.  Validate your results from question 1 by checking whether the average correlation of the numerical ratings in the 100 pairs is different from (higher?) than 100 randomly picked pairs of users from the full dataset.

## Movie recommendation

3.  As a first step, you will need to partition the ratings data into training, validation, and test sets. We recommend writing a script do this in advance, and saving the partitioned data for future use.
    This will reduce the complexity of your code down the line, and make it easier to generate alternative splits if you want to assess the stability of your implementation.

4.  Before implementing a sophisticated model, you should begin with a popularity baseline model as discussed in class. This should be simple enough to implement with some basic dataframe computations.
    Evaluate your popularity baseline (see below) before moving on to the next step.

5.  Your recommendation model should use Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.
    Be sure to thoroughly read through the documentation on the [pyspark.ml.recommendation module](https://spark.apache.org/docs/3.0.1/ml-collaborative-filtering.html) before getting started.
    This model has some hyper-parameters that you should tune to optimize performance on the validation set, notably: 
      - the *rank* (dimension) of the latent factors, and
      - the regularization parameter.

### Evaluation

Once you are able to make predictions—either from the popularity baseline or the latent factor model—you will need to evaluate accuracy on the validation and test data.
Scores for validation and test should both be reported in your write-up.
Evaluations should be based on predictions of the top 100 items for each user, and report the ranking metrics provided by spark.
Refer to the [ranking metrics](https://spark.apache.org/docs/3.0.1/mllib-evaluation-metrics.html#ranking-systems) section of the Spark documentation for more details.

The choice of evaluation criteria for hyper-parameter tuning is up to you, as is the range of hyper-parameters you consider, but be sure to document your choices in the final report.
As a general rule, you should explore ranges of each hyper-parameter that are sufficiently large to produce observable differences in your evaluation score.

If you like, you may also use additional software implementations of recommendation or ranking metric evaluations, but be sure to cite any additional software you use in the project.


### Using the cluster

Please be considerate of your fellow classmates!
The Dataproc cluster is a limited, shared resource. 
Make sure that your code is properly implemented and works efficiently. 
If too many people run inefficient code simultaneously, it can slow down the entire cluster for everyone.


## What to turn in

In addition to all of your code, produce a final report (no more than 5 pages), describing your implementation, answer to questions and evaluation results.
Your report should clearly identify the contributions of each member of your group. 
If any additional software components were required in your project, your choices should be described and well motivated here.  

Include a PDF of your final report through Brightspace.  Specifically, your final report should include the following details:

- Link to your group's GitHub repository
- List of top 100 most similar pairs (include a suitable estimate of their similarity for each pair), sorted by similarity
- A comparison between the average pairwise correlations between these highly similar pair and randomly picked pairs
- Documentation of how your train/validation splits were generated
- Any additional pre-processing of the data that you decide to implement
- Evaluation of popularity baseline
- Documentation of latent factor model's hyper-parameters and validation
- Evaluation of latent factor model

Any additional software components that you use should be cited and documented with installation instructions.

## Suggested Timeline

It will be helpful to commit your work in progress to the repository.
Toward this end, we recommend the following timeline to stay on track:

- [ ] 2025/04/21: data pre-processing, implementing the minHash algorithm
- [ ] **2025/04/28**: validation with correlations, checkpoint submission with similarity results.
- [ ] 2025/05/05: train/validation partitioning, popularity baseline model.
- [ ] 2025/05/12: Working latent factor model implementation. Then scale up to the full dataset in the remaining week until the final due date
- [ ] 2025/05/14: final project submission.  **NO EXTENSIONS ARE POSSIBLE PAST THIS DATE.** [this is the last day of the semester, 05/15 is commencement]
