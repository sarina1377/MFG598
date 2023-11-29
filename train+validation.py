import cv2
import numpy as np
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from collections import Counter



################### Part 1:
### In this code we extract the features from an image and then train KNN model:
### If you have already extracted the train data and want to modify only the KNN model simply comment below
# Load image and create a copy for drawing
file_path_test=r"C:\Users\sarin\Dropbox (ASU)\CEE\MFG 598\project\test.csv"
file_path_train=r"C:\Users\sarin\Dropbox (ASU)\CEE\MFG 598\project\train_video.csv"
image_path = input("Enter the path to the image: ")
img = cv2.imread(image_path)
img_draw = img.copy()
#Check if the image is loaded properly
if img is None:
      print("Error: Image not found. Please check the path.")

      #### image is imported propery, we draw the polygons:
else:
      for x in range(img.shape[0]):
          for y in range(img.shape[1]):
                  write_header = not os.path.exists(file_path_test)
                  ### writing the feature values of the test image to csv file:
                  with open(file_path_test, 'a', newline='') as datafile:
                      content = csv.writer(datafile)
                      if write_header:
                          content.writerow(['R', 'G', 'B', 'ExG', 'ExGR', 'CIVE'])
                      b,g,r = img[x,y]
                      exg = 2 * g - r - b
                      exgr = exg - (1.4 * r - g)
                      cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
                      content.writerow([r, g, b, exg, exgr, cive])
            

################### Part 2:
### We have now both train and test data, let's train the KNN model:


# Loading the training data from the provided file path.
train_data = pd.read_csv(file_path_train)  # Make sure to replace this with your actual file path.

# Preparing our data: we're separating the features (X) and the target label (y).
X = train_data.drop('Classification', axis=1)
y = train_data['Classification']

# Splitting our data into training and testing sets. This helps us evaluate our model later.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now, let's set up our KNN model. I used 3 
GS_KNN = KNeighborsClassifier(n_neighbors=3)
GS_KNN.fit(X_train, y_train)
# Time to predict!
y_pred = GS_KNN.predict(X_test)


# Now that the model is trained we'll see what is the vegetation prediction in our desired test img!
test_data = pd.read_csv(file_path_test)
test_predictions = GS_KNN.predict(test_data)


######## we can look at the classification predictions for the few first data
# Adding our predictions to the test data. This way, we can see them alongside the features.
#test_data_with_predictions = test_data.copy()
#test_data_with_predictions['Predicted_Classification'] = test_predictions
#print("\nFirst few predictions on test data:\n", test_data_with_predictions.head())



# Reporting the percentage for each classification
# Counting the occurrences of each classification
class_counts = Counter(test_predictions)
# Calculating the total number of predictions
total_predictions = len(test_predictions)
for classification, count in class_counts.items():
    percentage = (count / total_predictions) * 100
    print(f"{percentage:.2f}% of the image has the classification {classification}")


# Now we check the confusion matrix of the training model.
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)

# It's also important to know our model's overall accuracy, so let's print that too.
print('\nAccuracy of the KNN model is', accuracy_score(y_test, y_pred))

# Lastly, we'll visualize the confusion matrix. 
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
