# Exam Behavioral Anomaly Detection

## Project Overview

This project focuses on detecting behavioral anomalies in educational exam platforms through analysis of user interaction patterns. By leveraging mouse movements, keyboard actions, and temporal sequence data, the system identifies suspicious behavior that may support or indicate cheating or unusual activity during online exams and quizzes.

## Objective

Develop an unsupervised machine learning approach to detect anomalous behavior patterns in student exam sessions by analyzing:
- **Mouse Actions**: Movement patterns, clicks, and cursor trajectories
- **Keyboard Actions**: Typing speed, keystroke patterns, and input sequences
- **Temporal Data**: Time intervals between actions, session duration, and activity timing

## Key Features

- Behavioral pattern analysis from exam session recordings
- Unsupervised learning techniques for anomaly detection
- Classification of normal vs. suspicious exam behavior
- Support for mixed datasets (sessions with varying cheating indicators)

## Dataset

The project includes exam session data organized into:
- **Pure Normal Sessions**: Baseline sessions with normal student behavior
- **Mixed Sessions**: 5 Sessions with varying levels of anomalous activity

## Technical Approach

The project utilizes unsupervised learning methods to identify behavioral anomalies without requiring labeled training data. The approach includes:
- Feature extraction from interaction sequences
- Pattern recognition and clustering
- Anomaly scoring and classification

## Repository Structure

```
├── api.py                              # API for anomaly detection
├── Unsupervised-Approaches.ipynb       # Main analysis and model development
├── requirements.txt                    # Project dependencies
├── data/
│   ├── pure normal/                   # Normal student session data
│   └── mixed/                         # Sessions with mixed behaviors
└── readme.md                          # This file
```

## Getting Started

### Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

### Usage

1. **Analysis Notebook**: Open `Unsupervised-Approaches.ipynb` for exploratory analysis and model development
2. **API**: Use `api.py` for programmatic access to anomaly detection

## Output

The system provides:
- Anomaly scores for each user session
- Behavioral pattern classification
- Identification of suspicious activity indicators

## Future Enhancements

- Real-time anomaly detection during exam sessions
- Deep learning approaches for complex pattern recognition
- Cross-platform integration with learning management systems (LMS)
- Multi-modal analysis combining additional data sources
