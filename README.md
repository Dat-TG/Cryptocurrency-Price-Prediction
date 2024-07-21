# Cryptocurrency Price Prediction Dashboard

## Project Overview

This project is a web application for predicting cryptocurrency prices using LSTM models. The dashboard is built using Plotly Dash and provides visualization of actual and predicted closing prices for Bitcoin (BTC-USD), Ethereum (ETH-USD), and Cardano (ADA-USD).

Demo video here: [https://youtu.be/6Wz7xLmob4k?si=WuhEe0vOHSVZTMbm](https://youtu.be/6Wz7xLmob4k?si=WuhEe0vOHSVZTMbm)

The project is inspired by a [tutorial on stock price prediction](https://data-flair.training/blogs/stock-price-prediction-machine-learning-project-in-python/) by Data Flair.

## Student Information

- **Student ID:** 20120454
- **Name:** Cong-Dat Le
- **Module:** Advanced Topics in Software Development Technology
- **Instructor:** M.S. Van-Quy Tran, M.S. Duy-Quang Tran, M.S. Nguyen-Kha Do

## Technologies Used

- **Python**
- **Plotly Dash**
- **LSTM (Long Short-Term Memory) Model**
- **Pandas**
- **NumPy**
- **Keras**
- **TensorFlow**

## Project Structure

```plaintext
├── app.py                              # Main application script for the Dash web app.
├── cryptocurrency_prediction.ipynb     # Jupyter notebook for trainning the LSTM model.
├── cryptocurrency_data.ipynb           # Jupyter notebook for getting cryptocurrency price data from Yahoo Finance.
├── requirements.txt                    # List of Python dependencies required for the project.
├── csvdata                             # Directory containing CSV data files for cryptocurrencies.
│   ├── BTC-USD.csv                     # Historical price data for Bitcoin.
│   ├── ETH-USD.csv                     # Historical price data for Ethereum.
│   ├── ADA-USD.csv                     # Historical price data for Cardano.
│   └── all_data.csv                    # All data in one file.
├── saved_model.h5                      # Trained LSTM model file.
├── saved_model.keras                   # Trained LSTM model file.
├── README.md                           # Project overview, installation instructions, and usage guide.
└── LICENSE                             # This project is licensed under the MIT License
```

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/Dat-TG/Cryptocurrency-Price-Prediction.git
   cd Cryptocurrency-Price-Prediction
   ```

2. Create a virtual environment and activate it:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the dependencies:

   ```sh
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure your model file (`saved_model.h5`) and CSV data files are in the appropriate directories.

2. Run the Dash application:

   ```sh
   python app.py
   ```

3. Open your web browser and go to `http://127.0.0.1:8050/` to view the dashboard.

## Deployment

The project is deployed on [Render](https://render.com/) and can be accessed at [https://cryptocurrency-price-prediction.onrender.com/](https://cryptocurrency-price-prediction.onrender.com/).

To deploy the application to Render:

1. Ensure your project contains `requirements.txt`.

2. Push your code to a GitHub repository.

3. Create a new Web Service on Render and connect it to your GitHub repository.

4. Render will automatically detect the `requirements.txt` and use it to build and run your app.

## Usage

- The dashboard contains three tabs for Bitcoin (BTC-USD), Ethereum (ETH-USD), and Cardano (ADA-USD), and one tab for comparison high-low and volume of them.
- Each tab of each cryptocurrency displays:
  - A combined graph showing actual vs. predicted closing prices.
  - A scatter plot of actual closing prices.
  - A scatter plot of LSTM predicted closing prices.

## Acknowledgements

This project is part of the Advanced Topics in Software Development Technology module, guided by M.S. Van-Quy Tran, M.S. Duy-Quang Tran, and M.S. Nguyen-Kha Do.

The code is inspired by a tutorial on stock price prediction by Data Flair.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
