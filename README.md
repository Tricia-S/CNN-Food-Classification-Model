![image](https://github.com/user-attachments/assets/23bd053b-80c6-4d22-9086-17d501ab5e60)
# CNN-Food-Classification-Model

## Table of Contents
- [Overview](#overview)
- [Data Preprocessing and Data Loading](#data-preprocessing-and-data-loading)
- [Develop the Image Classification Models (at least TWO)](#develop-the-image-classification-models-at-least-two)
- [Evaluate models using Test images](#evaluate-models-using-test-images)
- [Summary](#summary)

## Overview
The project aims to apply classification model techniques to classify a specific set of 10 food items. It requires selecting and comparing multiple models, to determine which performs best in terms of accuracy and loss. The goal is not only to create a reliable model for classification but also to ensure that it can be validated and tested using real-world data. This project will involve a detailed analysis of the model's performance through metrics such as accuracy and loss curves and metrics like precision, recall, and F1-scores for the final 2 models chosen. The problem statement is to develop a food classification model to classify this set of 10 assigned food items (edamame, ice cream, bread pudding, grilled salmon, poutine, tiramisu, fish and chips, takoyaki, churros, and devilled eggs) as accurately as possible.

## Data Preprocessing and Data Loading
I preprocessed and loaded the data in a separate jupyter notebook file before I could start with modelling.  Firstly, I was trying to start by setting up the base and image directories. The base directory is base_dir and the image directory is image_dir. Essentially, this is so that we can define the base_dir is the current working directory and the image_dir directs to where the food images are stored. 
Next, I then created more directories but for data splits like train, validation, and test directories, train_dir, validation_dir, and test_dir respectively. Those new three directories I created in image_dir are for storing training, validation, and test images.  Ultimately this is just to create separate folders for training, validation, and test datasets to organize your data for model training.
After that, I can finally start to load and assigning the 10 different food categories I was assigned to, reading and then loading them from my 46.txt file to ‘food_list’. This is so that we can read food categories from a text file into a ‘food_list’ to know which images belong to which category
Next, I started with creating separate folders for each food category helps organize the training data. This structure is crucial for ensuring that each category's images are stored together, which makes it easier to manage and access the data during model training. And after that, the first 750 images of each food category are all copied into their respective folders in the training directory for the training dataset. And my understanding is that, copying the images helps me ensure that the original dataset is untouched and not modified, thus avoiding any altering of the source files.
Then I did the same for validation data and test data and they each have their own folders too. Validation with 200 images and test with 50. Specifically, in validation, I created directories for each food category in validation_dir and then I copied the next 200 images (indices 750 to 950) into these directories. The same goes for test_dir, where I created directories for each food category in test_dir and then copied the remaining 50 images (indices 950 to 1000) into these directories.

## Develop the Image Classification Models (at least TWO)
### Normal Models Tested for Model 1
![image](https://github.com/user-attachments/assets/15bc8ad9-511f-4696-ba87-c2bb881705d5)


### Pre-Trained Models Tested for Model 2
![image](https://github.com/user-attachments/assets/99783afe-5cfa-4d21-b8e3-fa9a23189688)

Model 1.1:
![image](https://github.com/user-attachments/assets/7f60d858-8a41-4bc4-8645-88d23e3f4d83)
![image](https://github.com/user-attachments/assets/35cfd263-102a-41f8-8124-eb98b15be5cd)

This was my very first model, built without any data augmentation, parameter tuning, or adjustments to the learning rate. My intention was to establish a baseline performance or kind of like a naïve model. As expected, the model showed severe overfitting, as clearly illustrated in this chart.
From the start, the validation accuracy remained constant after fewer than ten epochs, barely improving beyond its initial value. Similarly, the validation loss shot up around the same time, increasing dramatically with each epoch after training began. This sharp increase created a significant gap between the validation and training metrics. While the training accuracy and loss indicated a strong performance, these results were misleading because the model was just memorising the training data rather than learning patterns that could generalise to unseen examples.
This behaviour is a textbook definition of overfitting, where the model focuses solely on the training dataset. Ultimately, causing the validation performance to deteriorate, making the model ineffective for practical applications. This baseline result confirmed the need for strategies like regularisation, learning rate adjustments, or data augmentation to improve the model's generalisation ability and mitigate overfitting.

Model 1.2:
![image](https://github.com/user-attachments/assets/454532ec-6e23-4b9a-8132-17494f504c47)
![image](https://github.com/user-attachments/assets/f59ac166-25fc-4605-bf14-ed2716a9ace2)

In these charts, the tail end of the validation curves for both loss and accuracy are starting to pull away from the training curves. For accuracy, the validation curve is fluctuating below the training curve. For loss, the validation loss is starting to increase slightly, while the training loss keeps decreasing. This divergence suggests that overfitting is starting to happen, and the model is memorising the training data too much instead of learning patterns that generalise to the validation set.
In this model, the only difference between this and the chosen model for model 1 below is that I didn’t modify the data augmentation parameters to have additional settings like adjusting the ‘fill_mode’ to reflect or adjusting the rotations once again or adjusting the brightness of the images. And thus, this concludes that the model performance is definitely impacted by the data augmentation parameters.

Model 1.3 (chosen Model 1):
![image](https://github.com/user-attachments/assets/9d9a5bae-5ed1-44d8-909e-eaba82460235)
![image](https://github.com/user-attachments/assets/befe14ca-74c6-4146-b868-164fc2e775b6)
 
The charts show how well a model performs during training and validation over 100 epochs. The top chart is about accuracy, which kind of displays how often the model gets things right), while the bottom chart shows the loss how far off the model's predictions are from the true answers.
For the training and validation accuracy curve, the top chart, you can see the dots, the training accuracy, steadily going up as the model gets better at learning from the training data. The blue line, the validation accuracy, follows a similar pattern but is bumpier. This bumpiness is normal because validation checks are done on unseen data, so it's more variable. Towards the end, both training and validation accuracy seem to level off around 0.733. This means the model has learned quite a bit, but it’s probably hit its limit.
For the training and validation loss curve, the bottom chart, both the dots, the training loss, and the line, the validation loss, go down, which is what we want. Lower loss means the model is getting better at predicting correctly. The training loss decreases smoothly, but the validation loss has spikes. These spikes could mean that the model sometimes struggles with the unseen data, which potentially is due to noise. 
Overall, the model is learning well, it’s improving both in accuracy and lowering the loss. But the validation loss's spikes and the bumpy validation accuracy suggest the model might not generalise perfectly to new data. It’s not overfitting too badly though, since the validation metrics don’t get worse over time, which is a good sign.
Additionally, I’ve also modified the data augmentation parameters like fill_mode='reflect' and adjusting brightness, you’re helping the model focus on the important parts of the image while maintaining a realistic appearance of the food. The reflect fill mode is especially helpful because it avoids creating weird artifacts around the edges of images when you apply transformations. Adjusting brightness also ensures that darker or overly bright images don’t throw off the model. And the fact that my validation accuracy closely follows training accuracy shows that the augmented images are doing a good job of simulating real-world scenarios. 
Overall, since this model produced the best performing score with a low validation loss and a moderately high validation accuracy, while being able to keep the model from overfitting severely, is why I decided to choose this model for my Model 1. 

Model 1.4:
![image](https://github.com/user-attachments/assets/09c70c8d-bd17-4dfc-807b-67e19d566f29)
![image](https://github.com/user-attachments/assets/e86e2610-7be7-4deb-9743-7dfabe842104) 
 
I tried to improve the validation accuracy by making the model more complex. Specifically, I changed the last conv2d layer from 128 to 256. I also kept the learning rate, batch size, and the number of epochs the same, as well as the data augmentation, all parameters stayed the same as the parameters in the chosen model. 
Unfortunately, this approach led to a significant difference between the training and validation scores while the model trains over the epochs, with validation accuracy levelling off much earlier than training accuracy, indicating potential overfitting. This suggests that the model is performing well on training data but not generalizing well to new data which is the validation data. Hence, I could only infer that this tuning parameter wasn’t effective as the results as it led to the classic issue of when the model becomes too complex for the available data.

Model 1.5:
![image](https://github.com/user-attachments/assets/bb8a5a52-ef30-4d57-b85f-06eb6f235b14)
![image](https://github.com/user-attachments/assets/c7a85b91-97de-437a-9b41-25dd20e95a18)
 
For this model, I tried to improve the validation accuracy by making it more complex. Specifically, I increased the last Conv2D layer from 128 to 256. I kept the learning rate, batch size, and data augmentation code unchanged. Since I aimed to make the model more complex, I decided to reduce the number of epochs from 100 to 50, thinking it might help reduce overfitting. Unfortunately, that wasn’t the case. The training and validation accuracy curves clearly show that the model began to overfit around the 30-epoch mark. Similarly, the training and validation loss curves indicate signs of overfitting appearing a few epochs after the 20th epoch, in the early 20s. As expected, the accuracy scores of this model with 50 epochs were lower compared to the same model trained with 100 epochs. The accuracy score decreased noticeably. For these reasons, I ultimately decided not to choose this model.

