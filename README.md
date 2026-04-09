# 🔬 Customer Segmentation Lab

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [About](#-about)
- [Key Improvements](#key-improvements)
- [Quick Start](#-quick-start)
- [Project Dependencies](#-project-dependencies)
- [Features & Sections](#-features--sections)
- [Configuration](#️-configuration)
- [Project Structure](#-project-structure)
- [How to Use](#-how-to-use)
- [Algorithms & Methods](#-algorithms--methods)
- [Security Features](#-security-features)
- [Troubleshooting](#-troubleshooting)
- [User Feedback](#-user-feedback)
- [Deployment](#-deployment)
- [Performance Considerations](#-performance-considerations)
- [Contributing](#-contributing)
- [License](#-license)
- [Support](#-support)
- [Authors](#-authors)

---

## About

Welcome to the **Customer Segmentation Lab**, an advanced interactive application built with **Streamlit** for comprehensive customer data analysis and segmentation. This application empowers businesses to understand their customer base through machine learning, RFM analysis, and feature engineering, enabling data-driven marketing strategies and customer relationship management.

### Key Improvements

✨ **Enhanced Version** features:
- **Environment Variable Configuration** - Secure setup with `.env` file support
- **Advanced Feature Engineering** - Custom feature creation and transformation
- **Multiple Clustering Algorithms** - KMeans, DBSCAN, and Hierarchical Clustering
- **Comprehensive Model Comparison** - Side-by-side evaluation of clustering approaches
- **Advanced EDA** - Deep exploratory data analysis with advanced visualizations
- **Cluster Profiling** - Detailed analysis of customer segments
- **Improved Error Handling** - Fixed DataFrame type issues
- **Better UI/UX** - Modern dark sidebar with custom CSS styling

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CustomerSegmentation_Streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configurations if needed
   ```

4. **Run the application**
   ```bash
   streamlit run CustomerSegmentation_Streamlit.py
   ```

The app will open in your default browser at `http://localhost:8501`

---

## 📦 Project Dependencies

### Core Libraries

| Library | Purpose |
|---------|---------|
| **streamlit** | Web framework for interactive dashboards |
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical computing |
| **scikit-learn** | Machine learning and preprocessing |
| **plotly** | Interactive visualizations |
| **matplotlib** | Static plotting |
| **seaborn** | Statistical data visualization |
| **squarify** | Treemap visualization |
| **python-dotenv** | Environment variable management |

### install via requirements.txt
```
matplotlib
plotly
scikit-learn
streamlit
seaborn
squarify
import-ipynb
findspark
python-dotenv
```

---

## 🎯 Features & Sections

### 📋 Business Understanding
- Project overview and objectives
- Benefits of customer segmentation
- Business use cases and applications

### 📊 Data Understanding
- Sample data exploration or custom file upload
- Data preview and statistics
- Data quality assessment
- Initial descriptive analysis

### ⚙️ Data Preparation
- Missing value handling
- Data cleaning and validation
- Outlier detection and treatment
- Feature scaling and normalization
- Data quality reports

### 🧪 Feature Engineering
- RFM (Recency, Frequency, Monetary) analysis
- Custom feature creation
- Feature transformation and scaling
- Feature importance analysis
- Data preprocessing pipeline

### 📈 Advanced EDA
- Correlation matrix and heatmaps
- Distribution analysis
- Histogram and box plots
- Time-series trends
- Multivariate analysis

### 🤖 Modeling & Evaluation
- **Multiple Algorithms**:
  - KMeans Clustering
  - DBSCAN
  - Hierarchical (Agglomerative) Clustering
- **Evaluation Metrics**:
  - Elbow Method
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Score
- **Dimensionality Reduction**: PCA visualizations
- **Interactive Charts**: Scatter plots, 3D plots, treemaps

### 🏆 Model Comparison
- Side-by-side algorithm comparison
- Performance metrics comparison
- Optimal parameter recommendations
- Algorithm selection guidance

### 🎯 Cluster Profiling
- Segment characteristics
- Customer profiles per cluster
- Business insights and recommendations
- Segment statistics and distributions

### 🔮 Predict
- New customer cluster prediction
- RFM score calculation for new data
- Batch prediction support
- Results export (CSV download)
- Interactive form-based input

---

## ⚙️ Configuration

### Environment Variables

Configure via `.env` file (optional, defaults provided):

```env
# Streamlit Server Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_LOGGER_LEVEL=info

# Application Settings
DEBUG=false
LOG_LEVEL=INFO

# Data Paths
DATA_PATH=data
FEEDBACK_PATH=feedback.csv
MODEL_PATH=models
```

---

## 📁 Project Structure

```
CustomerSegmentation_Streamlit/
├── CustomerSegmentation_Streamlit.py   # Main application
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── .env.example                        # Environment variables template
├── .env                                # Local environment configuration
├── .gitignore                          # Git ignore patterns
├── data/                               # Sample datasets
│   ├── CDNOW_master.txt
│   └── CDNOW_sample.txt
└── [Dynamically created]
    ├── feedback.csv                    # User feedback storage
    ├── kmeans_model.pkl                # Exported model
    ├── CDNOW_master_new.txt           # Processed data
```

---

## 💡 How to Use

### Workflow

1. **Start Application**: Run `streamlit run CustomerSegmentation_Streamlit.py`

2. **Business Understanding**: Review project objectives and benefits

3. **Data Preparation**:
   - Upload your CSV file or use sample data
   - Review data statistics and quality

4. **Feature Engineering**:
   - Perform RFM analysis
   - Apply scaling and transformations
   - Review engineered features

5. **Modeling**:
   - Choose clustering algorithm
   - Select optimal parameters (K clusters, etc.)
   - Analyze silhouette scores and metrics

6. **Model Comparison** (Optional):
   - Compare multiple algorithms
   - Review performance metrics

7. **Cluster Profiling**:
   - Analyze segment characteristics
   - Review business insights

8. **Prediction**:
   - Input new customer data
   - Get cluster predictions
   - Export results

### Example Data Format

The application expects CSV data with the following columns:
```
Customer_id, day (YYYYMMDD), Quantity, Sales
1, 19970101, 1, 29.99
2, 19970112, 1, 39.99
```

---

## 📊 Algorithms & Methods

### Clustering Algorithms
- **KMeans**: Partitioning-based clustering with centroid optimization
- **DBSCAN**: Density-based clustering for finding arbitrary-shaped clusters
- **Hierarchical Clustering**: Agglomerative clustering with dendrograms

### Evaluation Metrics
- **Silhouette Score**: Measures how similar an object is to its own cluster (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Ratio of sum of within-cluster to between-cluster distances (lower is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster dispersion (higher is better)

### RFM Analysis
- **Recency**: How recently did the customer purchase?
- **Frequency**: How often did they purchase?
- **Monetary**: How much did they spend?

---

## 🔐 Security Features

- ✅ Environment variables for sensitive configuration
- ✅ `.gitignore` prevents accidental commits of `.env` and data files
- ✅ Session-based state management
- ✅ CSV feedback storage for audit trails
- ✅ Model persistence via pickle with local storage

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError`
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: `.env` file not loading
**Solution**: Verify `.env` file exists in the project root directory

### Issue: Port already in use
**Solution**: Change port in `.env`:
```
STREAMLIT_SERVER_PORT=8502
```

### Issue: Data not displaying
**Solution**: Check file format is CSV with expected columns and encoding is `latin-1`

---

## 📝 User Feedback

The application collects user feedback for continuous improvement. Feedback is stored in `feedback.csv` and includes:
- User comments
- Section visited
- Timestamp
- Rating (if provided)

---

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Configure secrets in Streamlit Cloud dashboard

### Heroku
1. Deploy using the provided `Procfile` and `setup.sh`
2. Set environment variables in Heroku dashboard

### Docker
Create a `Dockerfile` for containerized deployment.

---

## 📈 Performance Considerations

- Large datasets (>100K rows) may require optimization
- Feature engineering step scales with data size
- Model training time depends on cluster count and algorithm
- Consider data sampling for real-time EDA

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

This project is open source and available under the MIT License.

---

## ⭐ Support

If you found this project helpful, please star ⭐ the repository! Your support means a lot.

For questions, issues, or suggestions, please create an issue in the GitHub repository.

---

## 👥 Authors

- Original Project: [tieugem1997](https://github.com/tieugem1997)
- Enhanced Version: Continuous improvements and modern features added

---

**Last Updated**: April 2026
**Version**: 2.0 (Enhanced)
