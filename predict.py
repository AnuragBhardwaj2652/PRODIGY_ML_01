import joblib
import pandas as pd

# Load the saved model and transformers
model = joblib.load("model.joblib")
encoder = joblib.load("encoder.joblib")
imputer = joblib.load("imputer.joblib")

# Feature order must match training
features = [
    "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
    "FullBath", "YearBuilt", "KitchenQual", "Fireplaces"
]

# Get input from user
print("Enter the following house features:")
overall_qual = int(input("Overall Quality (1–10): "))
gr_liv_area = int(input("Above Ground Living Area (in sqft): "))
garage_cars = int(input("Garage Capacity (number of cars): "))
total_bsmt_sf = int(input("Total Basement Area (in sqft): "))
full_bath = int(input("Number of Full Bathrooms: "))
year_built = int(input("Year Built: "))
kitchen_qual = input("Kitchen Quality (Ex, Gd, TA, Fa): ")
fireplaces = int(input("Number of Fireplaces: "))

# Create DataFrame
input_data = pd.DataFrame([[
    overall_qual, gr_liv_area, garage_cars, total_bsmt_sf,
    full_bath, year_built, kitchen_qual, fireplaces
]], columns=features)

# Encode KitchenQual
input_data[["KitchenQual"]] = encoder.transform(input_data[["KitchenQual"]])

# Impute (if needed)
input_data = pd.DataFrame(imputer.transform(input_data), columns=features)

# Predict
predicted_price = model.predict(input_data)[0]
print(f"Predicted house price: ${predicted_price:,.2f}")