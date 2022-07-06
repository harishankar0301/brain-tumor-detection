# Brain Tumor Detection using Image Processing and Machine Learning Techniques

The application takes an MRI scan image as input and displays the output as ‘Tumor Positive’ or ‘Negative’. The users are required to upload the MRI scan image as input to the application. The input image is passed through <b>a series of processing steps</b>, where noise removal, sharpening etc. are performed. The image is then processed to <b>obtain important numerical features</b>, which are then passed to the model trained using Machine Learning techniques. It predicts the presence of a tumor based on its learning from the training data. This application can help users <b>become aware of the possibility of getting a brain tumor</b>, and be able to consult a doctor immediately for early diagnosis and treatment.

<b><u>Application Architecture:</u></b>

![ML arch - Flowchart](https://user-images.githubusercontent.com/54795291/177501369-1b3f6d4b-0202-4984-86e5-fb00848f7f65.png)


The overall architecture diagram of the project is shown above. First the input image is passed through various preprocessing steps such as binarization, stripping of the outer skull region, cleaning through erosion and dilation to enhance the image. The Region of Interest (ROI) is segmented from the enhanced image. Gray Level Co-occurrence Matrix technique is applied on the ROI to extract numerical features including such as energy, homogeneity, convex area etc. These series of steps are performed on all of the available data, and the resulting feature data is split into train and test data with 70% of the data used for training and the remaining for testing and evaluating the performance of the model. Feature normalization is also performed to improve the performance of the model. Various machine learning algorithms such as KNN, RandomForest, Logistic Regression and Support Vector Machine were used and their performance was compared to select the best performing model based on the performance on the test data.

<b>Modules:</b>
<ol>
<li>DATA PREPROCESSING AND FEATURE EXTRACTION</li>
<li>TRAINING AND TESTING THE MODEL</li>
<li>IMPLEMENTING THE MODEL IN THE REAL TIME SCENARIO</li>
<li>CREATION OF WEB APPLICATION AND DEPLOYMENT</li>
</ol>

<b>Libraries and Software Used:</b>
<ol>
<li>Programming Language: Python</li>
<li>Platform Used: Google Colab, Jupyter notebook for local development</li>
<li>Hosting Service: Microsoft Azure</li>
<li>Dataset Source: Kaggle</li>
<li>Libraries Used: Pandas, Matplotlib, Skimage, Sklearn, OpenCV, Numpy, Flask.</li>
</ol>


