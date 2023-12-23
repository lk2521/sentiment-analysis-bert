# Sentiment Analysis with Deep Learning using BERT
This project employs BERT (Bidirectional Encoder Representation from Transformers) for sentiment analysis, aiming to classify textual data into sentiment categories.

## Overview
The code encompasses several essential components:

### Exploratory Data Analysis and Preprocessing
This initial phase involves:

#### Data Loading: 
Reads and processes the data from the "Sentences_AllAgree.txt" file.
#### Data Cleaning: 
Handles any formatting issues, such as newline characters, and separates comments from their associated labels.
#### Data Representation: 
Stores the data in a Pandas DataFrame for better organization.

### Training-Validation Split
This section performs:

#### Data Split: 
Segregates the dataset into training and validation sets.
#### Stratification: 
Ensures balanced distribution of labels in both training and validation sets.

### Loading Tokenizer and Encoding Data
This stage involves:

#### Tokenization: 
Utilizes BERT's tokenizer to convert text data into tokenized sequences.
#### Padding: 
Adjusts sequences to a fixed length for uniform processing.
#### Conversion to Tensors: 
Converts encoded data into PyTorch tensors for training and validation.

### Setting up BERT Pretrained Model
This initializes:

#### BERT Model Configuration: 
Configures a BERT-based model for sequence classification.
#### Model Customization: 
Adjusts the model for the specific sentiment classification task.

### Creating Data Loaders
This step prepares:

#### Data Loader Creation: 
Constructs data loaders for efficient batching during model training and validation.

### Setting Up Optimizer and Scheduler
This section involves:

#### Optimizer Configuration: 
Configures the AdamW optimizer for model training.
#### Scheduler Initialization: 
Sets up a scheduler to adjust learning rates during training.

### Defining Performance Metrics
Defines:

#### F1 Score Calculation: 
Evaluates the model using the F1 score, which considers precision and recall.
#### Accuracy per Class: 
Measures accuracy for individual sentiment classes.

### Training Loop
Involves:

#### Model Training: 
Executes a training loop over multiple epochs, updating model parameters.
#### Validation and Evaluation: 
Assesses the model's performance on the validation set after each epoch.
#### Model Saving: 
Saves the fine-tuned BERT model for potential future use.

### Results
Following the completion of training and validation:

#### Training Loss: 
Provides insights into the loss function's performance during training.
#### Validation Loss: 
Indicates the model's loss on unseen validation data.
#### F1 Score (Weighted): 
Quantifies the model's performance using the weighted F1 score.

