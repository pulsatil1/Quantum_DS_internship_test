## Sentinel-2 image matching

Abstract: This repo contains an algorithm for matching satellite images. Notebook Dataset_creation.ipynb contains the code for obtaining a dataset based on data from Kaggle: [tap here](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine).
The Demo.ipynb file contains a demonstration of the image matching algorithm.
This algorithm uses a SIFT detector to detect key points, and a Brute-force matcher to match them.

Guide
a. We need to create a base directory (in my case it is called "Sentinel_image_matching"). Then put the files from the repository there.
b. Next, download the dataset from Kaggle and commit it to the current repository. You should have a "data" folder created. Or you can download an already processed dataset from Google Drive [tap here](https://drive.google.com/file/d/1dx1VJXx_vL9vDLWhbOyDRF200AsKXxVE/view?usp=sharing).
c. The base directory should now look something like this:

<pre>
 ├── data
 ├── train.py
 ├── test.py
 ├── Dataset_creation.ipynb
 ├── Demo.ipynb 
</pre>

d. If you downloaded a dataset from Kaggle, run the Dataset_creation.ipynb file to generate a dataset with smaller, unfiltered images in a new directory.
e. Run the train.py file to see the image keypoint matching algorithm for two images in action.
f. Specify the name of the desired image in the config.py file, and run the test.py file to see how the algorithm maps this image to all other images.