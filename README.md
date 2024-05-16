# ML in Real Estate: Analysis of Real Estate Prices in the New York Metropolitan Area

## Introduction
This project employs machine learning techniques to analyze real estate prices in the New York Metropolitan Area. The aim is to provide insights into the factors influencing property values and predict future price trends.

## Project Overview
This repository contains the code, data, and documentation for analyzing real estate prices using machine learning models. The key components of the project include data preprocessing, model training, and a web application for interactive data visualization.

## Project Report and Presentation
- **4.26Proposal.pdf**: Project First Proposal submitted on 4/26/2024.
- **Presentation_A_ML_Approach_to_the_Analysis_of_Real_Estate_Prices_in_the_New_York_Metropolitan_Area.pdf**: Project Final Presentation.
- **FinalReport_A_ML_Approach_to_the_Analysis_of_Real_Estate_Prices_in_the_New_York_Metropolitan_Area.pdf**: Project Final Report submitted on 5/15/2024.


## Installation
To set up this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/username/ML-in-Real-Estate.git
   cd ML-in-Real-Estate

2. **Usage:**
To run the main application, use the following command:
    ```sh
    streamlit run app.py

Use **New York, NY_sold_past365days.csv** for Streamlit app in your browser. This will start the web application where you can interact with the data and visualize the analysis results.

## Project Structure
- **app.py**: Main script to run the web application.
- **data/**: Directory containing the datasets used in the analysis.
- **data_preprocessing/**: Scripts for data cleaning and preprocessing.
- **models/**: Trained machine learning models and related scripts.
- **pages/**: HTML templates and static files for the web application.


## Data Files

- **New York, NY_sold_past365days.csv**: The raw data file obtained from [Realtor](https://www.realtor.com/) scraping.
- **housing_data.csv**: The file exported after preprocessing data from New York, NY_sold_past365days.csv.
- **test_data.csv**: The test dataset split from housing_data.csv for model training.
- **train_data.csv**: The training dataset split from housing_data.csv for model training.


## Data Collection Model
In the data scraping section, we referenced scraping model from [HomeHarvest](https://github.com/Bunsly/HomeHarvest) and used the following code to perform data scraping:
    
    from homeharvest import scrape_property
    # modify the parameters to your needs
    location = "New York, NY"
    listing_type = "sold" 
    past_days=365

    # Generate the filename based on the parameters
    filename = f"{location}_{listing_type}_past{past_days}days.csv"
    properties = scrape_property(
        location=location,
        listing_type=listing_type,
        past_days=past_days,
        
    )

    print(f"Number of properties: {len(properties)}")
    # delete the columns that are not needed
    columns_to_drop = [
        'property_url', 'mls', 'text', 'state', 
        'list_date', 'last_sold_date', 'unit',
        'mls_id', 'neighborhoods',
        'fips_code','hoa_fee','parking_garage','agent',
        'broker','broker_phone','nearby_schools','primary_photo','alt_photos'
    ]

    # drop the columns
    properties = properties.drop(columns=columns_to_drop)

    # save to a CSV without the index
    properties.to_csv(filename, index=False)
    print(properties.head(3))


## Contact Information
For any questions or inquiries, please contact us at email ph474@cornell.edu, zx324@cornell.edu, sc2745@cornell.edu, yl3692@cornell.edu, yz2947@cornell.edu.
