# DS-GA 1003 project: Fake review detection

This is a fake review detection system we are creating for NYU's DS-GA 1003
final project. The dataset we're using comes from Yelp and is described [here](https://worksheets.codalab.org/worksheets/0x33171fbfe67049fd9b0d61962c1d05ff).
The training, validation and test sets can also be found as `.csv` files on the
Codalab page.

## Getting the data

In order to run this modeling pipeline, you need to download the `.csv` files
from Codalab and place them in the `data/` directory first. Once that's done,
run the following Jupyter notebooks, in order:

1. `Kelsey/Pre_vectorization_feature_engineering.ipynb`
2. `Aren/FeatureTransformation.ipynb`
3. `Sid/ratings.ipynb`
4. `Kelsey/downsample.ipynb`
5. `Sid/vectorize-count.ipynb`
6. `Sid/concat-features.ipynb`

Once the features have been concatenated, the data will be in the `data/` folder.

At this point, if you want to run through our baseline model, you can do so in
`Sid/baseline-model.ipynb`.

**Collaborators:**

Kelsey Markey

Aren Dakessian

Guido Petri
