# ML-in-Real-Estate
A Machine Learning Approach to the Analysis of Real Estate Prices in the New York Metropolitan Area

# data collection demo -- haven't uploaded yet
New York, NY_sold_past365days.csv(10000rows)


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

# data processing -- nyc_house_preprocessing.ipynb (9981rows)
Handling missing data;
Data cleaning;
Type conversions and filtering;
Feature engineering;
Adding new features


# further steps:
# data visualization
Exploratory data analysis & data_visualization;
geo_data_visualization;
streamlit

# model training & test & develop