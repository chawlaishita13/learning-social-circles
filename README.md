# Group 7: Learning Social Circles
Hello, and welcome to our repository! 

## File Structure

```
Root Directory
├── data
|   ├── egonets.zip
|   ├── features.zip
|   ├── Training.zip
|   └── featureList.txt
├── boosting
│   ├── dataloader.py
│   └── main.py
├── JaccardSimilarity
│   ├── dataloader.py
|   ├── JaccardSimilarity.py
│   └── main.py
├── LogisticRegression
|   ├── dataloader.py
|   ├── main.py
|   └── utils.py
├── SVD
|   ├── dataloader.py
|   ├── SVD.py
|   └── main.py
├── feature_analysis.ipynb
├── utility_funcs.py
├── README.md
└── requirements.txt
```

## How to use
First, please install all dependencies. We recommend using a virtual environment, so you don't clutter your global python with these packages. To create a virtual environment, first install the venv python module with ```pip install venv```. After this, you can create a virtual environment with ```python -m venv {name of virtual environment}```, inserting the name you want to give this virtual environment where the {} are. After that, please activate the virtual environment, and then run ```pip install -r requirements.txt``` from the root directory of our repository. This will install all the necessary dependencies!

Once you've done that, please look in the ```data``` folder, and if the .zip files aren't already unzipped, please unzip them inside the ```data``` folder. If you chose to put the data elsewhere, please update the paths in the files, or the code will not function.

If you would like the seer the visualizations created for our presentation, please direct your attention to ```feature_analysis.ipynb```. This notebook contains the figures we used. If you want to use the individual modules, please check out the main.py files located in each of the model folders. You can either run these with no modification to see what they do, or use them as a starting point for your own implementation.

### Dependencies
* Numpy
* Pandas
* SK Learn
* Networkx
* Tqdm
* Xgboost
* Matplotlib
The complete list can be found in ```requirements.txt```.
