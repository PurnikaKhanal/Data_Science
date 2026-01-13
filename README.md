# Data Science Projects Portfolio

## Overview

This repository contains a comprehensive collection of data science programs and projects implemented in Python, designed to demonstrate proficiency across various domains of data science and machine learning. The projects range from foundational algorithms to advanced applications in natural language processing, recommendation systems, and financial forecasting.

## Table of Contents

- [Projects Overview](#projects-overview)
- [Installation](#installation)
- [Project Details](#project-details)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Learning Path](#learning-path)
- [Contributing](#contributing)
- [License](#license)

## Projects Overview

### 1. Machine Learning Classification Models

Implementation and comparison of various ensemble learning algorithms for classification tasks:

- **Decision Tree Classifier** (`1DecisionTreeClassifier.py`)
- **Random Forest Classifier** (`1RandomForestClassifier.py`)
- **Extra Trees Classifier** (`1ExtraTreesClassifier.py`)
- **Gradient Boosting Classifier** (`1GradientBoostingClassifier.py`)

These projects demonstrate the application of tree-based ensemble methods, hyperparameter tuning, model evaluation, and comparative analysis of different classification approaches.

### 2. Natural Language Processing (NLP)

#### Sentiment Analysis (`2SentimentAnalysis.py`, `2XSentimentAnalysis.py`)

Advanced sentiment analysis implementations utilizing natural language processing techniques to classify text data into positive, negative, or neutral categories. These projects showcase:

- Text preprocessing and tokenization
- Feature extraction from textual data
- Model training for sentiment classification
- Performance evaluation and optimization

### 3. Recommendation Systems (`3RecommendationSystem.py`)

Implementation of collaborative filtering and content-based recommendation algorithms. This project demonstrates:

- User-item interaction modeling
- Similarity computation techniques
- Recommendation generation algorithms
- Evaluation metrics for recommendation quality

### 4. Time Series Analysis and Forecasting

#### Stock Price Prediction (`4StockPricePrediction.py`)

A comprehensive financial forecasting system that leverages machine learning for stock market prediction. Features include:

- Historical stock data analysis (`stock.csv`)
- Time series feature engineering
- Predictive modeling with regression techniques
- Visualization of predictions and trends
- Model performance evaluation

## Installation

### Prerequisites

Ensure you have Python 3.7 or higher installed on your system. Also, a few dependencies are not available for any python versions after Python 3.11 so you may need to setup your environment accordingly. You can verify your Python installation by running:

```bash
python --version
```

### Setting Up the Environment

1. **Clone the Repository**

```bash
git clone https://github.com/PurnikaKhanal/Data_Science.git
cd Data_Science
```

2. **Create a Virtual Environment (Recommended)**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install Required Dependencies**

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the core dependencies manually:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn nltk textblob
```

## Dependencies

The projects utilize the following Python libraries:

### Core Data Science Stack
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and utilities

### Visualization
- **Matplotlib**: Static plotting and visualization
- **Seaborn**: Statistical data visualization

### Natural Language Processing
- **NLTK**: Natural Language Toolkit for text processing
- **TextBlob**: Simplified text processing and sentiment analysis

### Additional Libraries
- **SciPy**: Scientific computing utilities
- **Statsmodels**: Statistical modeling (for time series analysis)

## Usage

Each Python file is designed to be executed independently. Navigate to the project directory and run:

```bash
python <filename>.py
```

### Example: Running the Stock Price Prediction

```bash
python 4StockPricePrediction.py
```

### Example: Running Sentiment Analysis

```bash
python 2SentimentAnalysis.py
```

## Project Details

### Machine Learning Classification Suite

#### Decision Tree Classifier
A single decision tree model that learns decision rules from training data. Ideal for understanding the fundamentals of tree-based learning.

**Key Concepts:**
- Information gain and entropy
- Tree pruning and regularization
- Feature importance analysis

#### Random Forest Classifier
An ensemble method that constructs multiple decision trees and aggregates their predictions for improved accuracy and reduced overfitting.

**Key Concepts:**
- Bootstrap aggregating (bagging)
- Random feature selection
- Out-of-bag error estimation

#### Extra Trees Classifier
An extremely randomized trees algorithm that introduces additional randomness in tree construction for enhanced model diversity.

**Key Concepts:**
- Randomized splitting
- Variance reduction
- Computational efficiency

#### Gradient Boosting Classifier
A sequential ensemble technique that builds models iteratively, with each new model correcting errors made by previous ones.

**Key Concepts:**
- Boosting methodology
- Learning rate optimization
- Sequential model improvement

### Natural Language Processing Projects

#### Sentiment Analysis
Analyzes textual data to determine the emotional tone and polarity of the content. Applications include social media monitoring, customer feedback analysis, and opinion mining.

**Implementation Highlights:**
- Text cleaning and preprocessing
- Vectorization techniques (TF-IDF, Count Vectorization)
- Classification model training and evaluation
- Real-world application scenarios

### Recommendation Systems

Implements collaborative filtering algorithms to provide personalized recommendations based on user behavior and preferences.

**Implementation Highlights:**
- User-based collaborative filtering
- Item-based collaborative filtering
- Matrix factorization techniques
- Similarity metrics (cosine similarity, Pearson correlation)

### Financial Forecasting

#### Stock Price Prediction
Utilizes historical stock market data to forecast future price movements using regression and time series analysis techniques.

**Implementation Highlights:**
- Data preprocessing and feature engineering
- Technical indicator calculation
- Model training with temporal validation
- Prediction visualization and analysis
- Risk assessment and confidence intervals

## Learning Path

This repository is structured to facilitate learning from fundamental to advanced concepts:

### Beginner Level
1. Start with `1DecisionTreeClassifier.py` to understand basic classification
2. Progress to `1RandomForestClassifier.py` for ensemble methods

### Intermediate Level
3. Explore `1ExtraTreesClassifier.py` and `1GradientBoostingClassifier.py` for advanced ensemble techniques
4. Dive into `2SentimentAnalysis.py` for NLP fundamentals

### Advanced Level
5. Work with `3RecommendationSystem.py` for complex algorithm implementation
6. Complete `4StockPricePrediction.py` for time series and financial analysis

## Best Practices Demonstrated

- **Code Organization**: Modular and well-structured Python scripts
- **Data Handling**: Efficient data loading, preprocessing, and transformation
- **Model Evaluation**: Comprehensive metrics and cross-validation techniques
- **Visualization**: Clear and informative plots for data exploration and results
- **Documentation**: Inline comments and clear variable naming

## Future Enhancements

### Planned Projects
- Deep Learning implementations (Neural Networks, CNNs, RNNs)
- Advanced NLP with transformer models (BERT, GPT)
- Computer vision applications
- Reinforcement learning algorithms
- Big Data processing with PySpark
- Deployment pipelines with Flask/FastAPI

### Improvements
- Jupyter Notebook versions for interactive exploration
- Comprehensive unit testing
- Hyperparameter optimization automation
- Docker containerization for reproducibility
- CI/CD pipeline integration

## Contributing

Contributions are welcome and encouraged. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-project`)
3. Commit your changes with descriptive messages
4. Push to your branch (`git push origin feature/new-project`)
5. Open a Pull Request with a detailed description

### Contribution Guidelines

- Ensure code follows PEP 8 style guidelines
- Include docstrings and comments for clarity
- Add appropriate error handling
- Update documentation as needed
- Test your code thoroughly before submitting

## Datasets

- `stock.csv`: Historical stock market data for financial forecasting projects
- Additional datasets may be downloaded programmatically within individual scripts

## Resources and References

### Recommended Learning Materials
- **Books**: "Hands-On Machine Learning with Scikit-Learn and TensorFlow" by Aurélien Géron
- **Documentation**: [Scikit-learn Official Documentation](https://scikit-learn.org/)
- **Courses**: Andrew Ng's Machine Learning Course, Fast.ai courses

### Useful Links
- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [NLTK Documentation](https://www.nltk.org/)

## License

This project is available for educational and non-commercial purposes.

## Contact

**Developer**: Purnika Khanal  

For questions, suggestions, or collaboration opportunities, please open an issue on the GitHub repository.

## Acknowledgments

This collection represents a journey through various domains of data science, implementing algorithms and techniques learned from academic coursework, online courses, and hands-on experimentation. Special thanks to the open-source community for providing excellent libraries and resources that make data science accessible to everyone.

---

**Note**: These projects are designed for educational purposes and demonstration of data science concepts. For production applications, additional considerations regarding scalability, robustness, and deployment best practices should be implemented.
