# AI-comment-classifier
An ML-powered classifier to distinguish between human-generated and AI-generated comments on social media, using machine learning techniques for accurate detection and analysis

## Overview
This project is designed to classify social media comments as either human-generated or AI-generated. Using machine learning techniques, the classifier aims to accurately distinguish between the two types of content, focusing on key text features and model evaluation metrics.

## Features
- **Data Preprocessing**: Cleans and prepares text data for model training.
- **Modeling**: Implements a logistic regression model for binary classification.
- **Evaluation**: Utilizes precision, recall, F1-score, and other metrics to evaluate model performance.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AI-Comment-Classifier.git
   cd AI-Comment-Classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare your dataset:** 
   - Ensure your data is in a CSV format with the appropriate labels (human-generated or AI-generated).

2. **Run the model:**
   - Use the provided scripts to train the model on your dataset.
   - Evaluate the results using the included evaluation functions.

3. **Example Command:**
   ```bash
   python train_model.py --data data/comments.csv
   ```

## Project Structure
- **data/**: Sample datasets used for training and testing.
- **src/**: Contains the main code files, including preprocessing, modeling, and evaluation scripts.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and model experimentation.
- **app.py**: Flask application python file
- **.pkl files**: Needed for flask application to use trained model
- **templates/index.html**: Base HTML UI file
- **requirements.txt**: List of dependencies required to run the project.

## Demo

![Human comment](https://github.com/user-attachments/assets/04bc7448-ac51-4955-8ae6-a3d9f46e29b1)
![AI generated comment](https://github.com/user-attachments/assets/8a5179f4-e1db-412f-8d02-fa0fbd954650)
![image](https://github.com/user-attachments/assets/9a5471da-1678-4785-b1bf-fb71e41895d6)

## Contributing
If you would like to contribute to this project, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions or suggestions, please contact [Salil Apte](mailto:salil.apte99!gmail.com) / [Surya Ammisetti](sammisetti1@student.gsu.edu) / [Shrinidhi Sivakumar](ssivakumar5@student.gsu.edu)
