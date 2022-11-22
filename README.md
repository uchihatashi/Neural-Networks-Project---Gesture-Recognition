# Gesture Recognition

# **Problem Statement**

Imagine you are working as a data scientist at a home electronics company which manufactures state of the art **smart televisions**. You want to develop a cool feature in the smart-TV that can **recognise five different gestures** performed by the user which will help users control the TV without using a remote. Let's have professor Raghavan introduce you to the problem statement:

The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:

- Thumbs up: Increase the volume
- Thumbs down: Decrease the volume
- Left swipe: 'Jump' backwards 10 seconds
- Right swipe: 'Jump' forward 10 seconds
- Stop: Pause the movie

Each video is a sequence of 30 frames (or images). In the next couple of lectures, our subject matter expert Snehansu will walk you through the structure of the dataset.

## **Understanding the Dataset**

The training data consists of a few hundred videos categorised into one of the five classes. Each video (typically 2-3 seconds long) is divided into a **sequence of 30 frames(images)**. These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use.

In the following lecture, Snehansu will walk you through the problem statement and the dataset in detail.

The data is in a zip file. The zip file contains a 'train' and a 'val' folder with two CSV files for the two folders. These folders are in turn divided into subfolders where each subfolder represents a video of a particular gesture. Each subfolder, i.e. a video, contains 30 frames (or images). Note that all images in a particular video subfolder have the same dimensions but different videos may have different dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 (depending on the webcam used to record the videos). Hence, you will need to do some pre-processing to standardise the videos.

Each row of the CSV file represents one video and contains three main pieces of information - the name of the subfolder containing the 30 images of the video, the name of the gesture and the numeric label (between 0-4) of the video.

Your task is to train a model on the 'train' folder which performs well on the 'val' folder as well (as usually done in ML projects). We have withheld the test folder for evaluation purposes - your final model's performance will be tested on the 'test' set.

To get started with the model building process, you first need to get the data on your storage.

In order to get the data on the storage, perform the following steps in order

1. Open the terminal
2. go down [https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL](https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL)
3. unzip Project\_data.zip

Now that you have got the data on the storage, let's look at the different choices of architectures you can use.



### Conclusion:

**Here are all the models with their respective accuracy and number of parameters**

| Model No. | Model Type      | No. parameters | Valadation Acc(%) | Tranning Acc(%) | Input Parameter                                                                                         |
| --------- | --------------- | -------------- | ----------------- | --------------- | ------------------------------------------------------------------------------------------------------- |
| 0         | 3D CNN          | 9,00,805       | 19                | 88.84           | HxW 120, frame=30, batch\_size=55, epoch=20                                                             |
| 1         | 3D CNN          | 19,67,813      | 22                | 74.81           | HxW 120, frame=30, batch\_size=55, epoch=20                                                             |
| 2         | 3D CNN          | 19,67,813      | 21                | 92.91           | HxW 120, frame=30, batch\_size=55, epoch=15, dropout=0.5                                                |
| 3         | 3D CNN          | 17,62,613      | 68                | 83.26           | HxW 120, frame=30, batch\_size=55, epoch=25, dropout=0.5, filter=2                                      |
| 4         | 3D CNN          | 25,56,533      | 73                | 91.86           | HxW 120, frame=20, batch\_size=30, epoch=25, filter=3, dense\_neurons=256                               |
| 5         | 3D CNN          | 25,56,533      | 21                | 48.72           | HxW 120, frame=20, batch\_size=30, epoch=25, dropout=0.5, filter=3, dense\_neurons=256                  |
| 5.1       | 3D CNN          | 25,56,533      | 25                | 85.67           | HxW 120, frame=20, batch\_size=30, epoch=25, dropout=0.25, filter=3, dense\_neurons=256                 |
| 6         | 3D CNN          | 9,08,645       | 74                | 88.99           | HxW 120, frame=16, batch\_size=20, epoch=20, dropout=0.25, filter=2, dense\_neurons=128                 |
| 7         | 3D CNN          | 4,94,981       | 74                | 82.35           | HxW 120, frame=16, batch\_size=20, epoch=20, dropout=0.25, filter=2, dense\_neurons=64                  |
| 8         | CNN-LSTM        | 16,56,453      | 81                | 88.39           | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.25, filter=3, dense\_neurons=128, lstm\_cell=128 |
| 8.1       | CNN-GRU         | 1,346,405      | 75                | 93.21           | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.25, filter=3, dense\_neurons=128, GRU\_cell=128  |
| 9         | 3D CNN          | 19,66,309      | 79                | 71.87           | HxW 120, frame=20, batch\_size=20, epoch=25, dropout=0.5, filter=3, dense\_neurons=256                  |
| 10        | 3D CNN          | 17,61,109      | 48                | 68.7            | HxW 120, frame=16, batch\_size=30, epoch=25, dropout=0.5, filter=2, dense\_neurons=256                  |
| 11        | 3D CNN          | 25,54,549      | 83                | 70.21           | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.5, filter=2, dense\_neurons=256                  |
| 12        | 3D CNN          | 25,54,549      | 78                | 92.76           | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.25, filter=2, dense\_neurons=256                 |
| 13        | 3D CNN          | 9,09,637       | 88                | 92.38           | HxW 120, frame=30, batch\_size=20, epoch=25, dropout=0.25, filter=3,2, dense\_neurons=128               |
| 13.1      | 3D CNN          | 9,07,733       | 85                | 91.1            | HxW 120, frame=16, batch\_size=20, epoch=35, dropout=0.25, filter=2, dense\_neurons=128                 |
| **13.2**  | **3D CNN**      | **9,09,637**   | **94**            | **91.7**        | **HxW 120, frame=16, batch\_size=20, epoch=35, dropout=0.25, filter=3,2, dense\_neurons=128**           |
| 13.3      | 3D CNN          | 9,09,637       | 66                | 93.06           | HxW 120, frame=16, batch\_size=32, epoch=35, dropout=0.25, filter=3,2, dense\_neurons=128               |
| 14        | 3D CNN          | 4,94,245.00    | 81                | 85.9            | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.25, filter=2, dense\_neurons=64                  |
| 15        | CNN-GRU         | 25,57,413      | 82                | 99.55           | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.25, filter=3, dense\_neurons=128, GRU\_cell=128  |
| 15.1      | CNN-LSTM        | 33,76,357      | 78                | 99.4            | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.25, filter=3, dense\_neurons=128, LSTM\_cell=128 |
| 16        | TL-LSTM         | 3,840,453      | 72                | 99.55           | HxW 120, frame=16, batch\_size=10, epoch=20, dropout=0.25, filter=3, dense\_neurons=128, LSTM\_cell=128 |
| 16.1      | TL-LSTM         | 35,16,229      | 77                | 99.1            | HxW 120, frame=16, batch\_size=10, epoch=20, dropout=0.25, filter=3, dense\_neurons=64, LSTM\_cell=64   |
| 17        | TL-GRU          | 36,93,253      | 74                | 99.85           | HxW 120, frame=16, batch\_size=10, epoch=20, dropout=0.25, filter=3, dense\_neurons=128, GRU\_cell=128  |
| 18        | TL-GRU          | 34,46,725      | 70                | 99.4            | HxW 120, frame=16, batch\_size=10, epoch=20, dropout=0.25, filter=3, dense\_neurons=64, GRU\_cell=64    |
| 19        | TL-GRU-Non\_AUG | 3,446,725      | 71                | 99.25           | HxW 120, frame=16, batch\_size=10, epoch=20, dropout=0.25, filter=3, dense\_neurons=64, GRU\_cell=64    |
| 20        | TL-GRU-B2       | 83,81,950      | 16                | 20              | HxW 120, frame=16, batch\_size=10, epoch=20, dropout=0.25, filter=3, dense\_neurons=128, GRU\_cell=128  |


### So, here we have concluded that **Model no 13.2** gave the best accuracy score both for Validation(94%) and training (91.7%).
