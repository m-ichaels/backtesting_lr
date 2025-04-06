import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from stock_utils import get_data
import os
import pandas as pd
from lr_inference import LR_v1_predict

def plot_stock_price(stock='GBPUSD=X', start_date=datetime(2024, 1, 1), end_date=datetime(2025, 1, 1), n=10):
    """
    Fetches stock data and plots the closing price against time.
    """
    if start_date is None:
        start_date = datetime.today() - timedelta(days=365)  # Default to past year
    if end_date is None:
        end_date = datetime.today()
    
    # Fetch data
    data, idxs_with_mins, idxs_with_maxs = get_data(stock, start_date, end_date, n)
    # Plot data
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label=f'{stock} Closing Price', color='black')
    plt.scatter(data['Date'].iloc[idxs_with_mins], data['Close'].iloc[idxs_with_mins], color='green', label='Local Minima', marker='o')
    plt.scatter(data['Date'].iloc[idxs_with_maxs], data['Close'].iloc[idxs_with_maxs], color='red', label='Local Maxima', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.title(f'{stock} Closing Price Over Time')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)

    # Save the plot in the "results" folder
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    plot_path = os.path.join(results_folder, "data_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    print(idxs_with_mins)
    print(idxs_with_maxs)
    print(len(data))

    return data, idxs_with_mins, idxs_with_maxs

def create_prediction_dataframe(data, stock, threshold):
    prediction_df = data.copy()
    prediction_df['Prediction'] = None
    prediction_df['Prediction_Thresholded'] = None

    for i in range(len(prediction_df)):
        # Ensure Date is a datetime object before adding timedelta
        current_date = prediction_df['Date'].iloc[i]
        
        # If Date is not already a datetime object, convert it
        if not isinstance(current_date, datetime):
            try:
                # Try to parse it as a datetime if it's a string
                current_date = pd.to_datetime(current_date)
            except:
                print(f"Could not convert date at index {i}. Skipping.")
                continue
        
        period_start = current_date - timedelta(days=40)
        period_end = current_date + timedelta(days=1)
        
        try:
            prediction, prediction_thresholded, close_price = LR_v1_predict(
                stock, period_start, period_end, threshold
            )
            
            # Store the prediction for this date
            prediction_df.loc[i, 'Prediction'] = prediction
            prediction_df.loc[i, 'Prediction_Thresholded'] = prediction_thresholded
        except Exception as e:
            print(f"Error predicting for date {period_start}: {e}")

    return prediction_df

if __name__ == "__main__":
    stock = 'GBPUSD=X'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 1, 1)
    n = 10
    threshold = 0.92
    
    # Get stock data and plot
    data, idxs_with_mins, idxs_with_maxs = plot_stock_price(stock, start_date, end_date, n)
    
    # Create prediction dataframe - Fixed the for loop error here
    prediction_data = create_prediction_dataframe(data, stock, threshold)
    
    # Display the first few rows
    print("\nPrediction Data Preview:")
    print(prediction_data)

    # --- Step 3: Replot with Predictions Overlaid ---
    print("\nReplotting data with thresholded predictions...")
    plt.figure(figsize=(14, 7)) # Create a new figure

    # Plot original data and min/max points again
    plt.plot(data['Date'], data['Close'], label=f'{stock} Closing Price', color='black', linewidth=1.5, zorder=2)

    # Check if indices are valid before plotting scatter points
    valid_mins = [idx for idx in idxs_with_mins if idx < len(data)]
    valid_maxs = [idx for idx in idxs_with_maxs if idx < len(data)]

    if valid_mins:
        plt.scatter(data['Date'].iloc[valid_mins], data['Close'].iloc[valid_mins], color='lime', label='Local Minima', marker='^', s=100, edgecolors='black', zorder=5)
    if valid_maxs:
        plt.scatter(data['Date'].iloc[valid_maxs], data['Close'].iloc[valid_maxs], color='red', label='Local Maxima', marker='v', s=100, edgecolors='black', zorder=5)

        # Plot the thresholded prediction
        # Ensure 'Prediction_Thresholded' is numeric for plotting, convert Nones/NAs if necessary
    prediction_data['Prediction_Thresholded_Numeric'] = pd.to_numeric(prediction_data['Prediction_Thresholded'], errors='coerce')
    plt.plot(prediction_data['Date'], prediction_data['Prediction_Thresholded_Numeric'], label=f'Thresholded Prediction ({threshold})', color='blue', linestyle='--', marker='.', markersize=4, zorder=3) # Added marker

        # Add plot details
    plt.xlabel('Date')
    plt.ylabel('Price') # Generic Y-axis label
    plt.title(f'{stock} Price Over Time with Thresholded Predictions ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

    prediction_data['Is_Local_Min'] = False

# Check if idxs_with_mins is valid and contains indices

if idxs_with_mins is not None and len(idxs_with_mins) > 0:
    # Get valid indices that are within the bounds of the DataFrame
    valid_min_indices = [idx for idx in idxs_with_mins if idx < len(prediction_data)]

    if valid_min_indices:
        # Use .iloc to set 'Is_Local_Min' to True for the identified rows
        # Ensure you are targeting the correct column index if using iloc for assignment
        is_local_min_col_idx = prediction_data.columns.get_loc('Is_Local_Min')
        prediction_data.iloc[valid_min_indices, is_local_min_col_idx] = True
        print(f"Marked {len(valid_min_indices)} rows as local minima based on idxs_with_mins.")
    else:
        print("Warning: None of the provided local minima indices were within the DataFrame bounds.")
else:
    print("Warning: No local minima indices (idxs_with_mins) were provided or found.")


# --- Step 2: Ensure 'Prediction' column is numeric ---
# Convert 'Prediction' to numeric, setting errors='coerce' will turn non-numeric values into NaN
prediction_data['Prediction_Numeric'] = pd.to_numeric(prediction_data['Prediction'], errors='coerce')

# --- Step 3: Apply the filtering conditions ---
# Condition 1: The row is a local minimum ('Is_Local_Min' is True)
# Condition 2: The numeric prediction value is greater than the threshold
# Condition 3: The numeric prediction value is not NaN (implicitly handled by > comparison, but explicit check is clearer)

condition = (
    (prediction_data['Is_Local_Min'] == True) &
    (prediction_data['Prediction_Numeric'].notna()) &
    (prediction_data['Prediction_Numeric'] > threshold)
)

# Create the 'accurate' DataFrame by applying the combined condition
accurate = prediction_data[condition].copy() # Use .copy() to avoid SettingWithCopyWarning

# --- Step 4: Display Results ---
print(f"\nCreated 'accurate' DataFrame with {len(accurate)} rows.")

if not accurate.empty:
    print("Preview of 'accurate' DataFrame:")
    # Optionally drop the temporary numeric column before display/use
    print(accurate.drop(columns=['Prediction_Numeric']).head())
else:
    print("'accurate' DataFrame is empty.")

output_folder = "results"
if not os.path.exists(output_folder):
    try:
        os.makedirs(output_folder)
        print(f"Created output directory: ./{output_folder}")
    except OSError as e:
        print(f"Error creating directory ./{output_folder}: {e}")
        output_folder = "."
        print(f"Attempting to save in current directory: {os.getcwd()}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_filename = f"prediction_data_{timestamp}.xlsx"
excel_filepath = os.path.join(output_folder, excel_filename)

# Add 'Is_Local_Max' column to track local maxima
prediction_data['Is_Local_Max'] = False

# Check if idxs_with_maxs is valid and contains indices
if idxs_with_maxs is not None and len(idxs_with_maxs) > 0:
    # Get valid indices that are within the bounds of the DataFrame
    valid_max_indices = [idx for idx in idxs_with_maxs if idx < len(prediction_data)]

    if valid_max_indices:
        # Use .iloc to set 'Is_Local_Max' to True for the identified rows
        is_local_max_col_idx = prediction_data.columns.get_loc('Is_Local_Max')
        prediction_data.iloc[valid_max_indices, is_local_max_col_idx] = True
        print(f"Marked {len(valid_max_indices)} rows as local maxima based on idxs_with_maxs.")
    else:
        print("Warning: None of the provided local maxima indices were within the DataFrame bounds.")
else:
    print("Warning: No local maxima indices (idxs_with_maxs) were provided or found.")

# Define sell threshold
sell_threshold = 0.08

# Ensure 'Prediction' column is numeric for comparison
if 'Prediction_Numeric' not in prediction_data.columns:
    prediction_data['Prediction_Numeric'] = pd.to_numeric(prediction_data['Prediction'], errors='coerce')

# Create sell condition: Is a local maximum AND prediction is below sell threshold
sell_condition = (
    (prediction_data['Is_Local_Max'] == True) &
    (prediction_data['Prediction_Numeric'].notna()) &
    (prediction_data['Prediction_Numeric'] < sell_threshold)
)

# Create the 'sell_accurate' DataFrame by applying the combined condition
sell_accurate = prediction_data[sell_condition].copy()

# Display Results
print(f"\nCreated 'sell_accurate' DataFrame with {len(sell_accurate)} rows.")

if not sell_accurate.empty:
    print("Preview of 'sell_accurate' DataFrame:")
    print(sell_accurate.drop(columns=['Prediction_Numeric']).head())
else:
    print("'sell_accurate' DataFrame is empty.")

# Analysis of alignment between predictions and local extrema
print("\n==== PREDICTION ALIGNMENT ANALYSIS ====")

# ---- BUYING OPPORTUNITY ANALYSIS (prediction > threshold vs loc_min) ----
print("\n1. BUYING OPPORTUNITIES ANALYSIS:")

# Count cases where prediction > threshold but NOT at local minimum
high_prediction_not_min = (
    (prediction_data['Prediction_Numeric'] > threshold) & 
    (prediction_data['Is_Local_Min'] == False) &
    (prediction_data['Prediction_Numeric'].notna())
)
count_high_pred_not_min = high_prediction_not_min.sum()

# Count cases where is local minimum but prediction <= threshold
min_without_high_prediction = (
    (prediction_data['Is_Local_Min'] == True) &
    ((prediction_data['Prediction_Numeric'] <= threshold) | 
     (prediction_data['Prediction_Numeric'].isna()))
)
count_min_without_high_pred = min_without_high_prediction.sum()

# Count total high predictions and total minima for reference
total_high_predictions = (prediction_data['Prediction_Numeric'] > threshold).sum()
total_minima = prediction_data['Is_Local_Min'].sum()

# Count successful alignments (both conditions met)
accurate_count = len(accurate)

print(f"Total points with prediction > {threshold}: {total_high_predictions}")
print(f"Total local minima points: {total_minima}")
print(f"Points where prediction > {threshold} but NOT at local minimum: {count_high_pred_not_min}")
print(f"Local minima points without prediction > {threshold}: {count_min_without_high_pred}")
print(f"Successful alignments (prediction > {threshold} AT local minimum): {accurate_count}")

if total_high_predictions > 0:
    precision = accurate_count / total_high_predictions * 100
    print(f"Precision: {precision:.2f}% of high predictions are at local minima")

if total_minima > 0:
    recall = accurate_count / total_minima * 100
    print(f"Recall: {recall:.2f}% of local minima have high predictions")

# ---- SELLING OPPORTUNITY ANALYSIS (prediction < sell_threshold vs loc_max) ----
print("\n2. SELLING OPPORTUNITIES ANALYSIS:")

# Count cases where prediction < sell_threshold but NOT at local maximum
low_prediction_not_max = (
    (prediction_data['Prediction_Numeric'] < sell_threshold) & 
    (prediction_data['Is_Local_Max'] == False) &
    (prediction_data['Prediction_Numeric'].notna())
)
count_low_pred_not_max = low_prediction_not_max.sum()

# Count cases where is local maximum but prediction >= sell_threshold
max_without_low_prediction = (
    (prediction_data['Is_Local_Max'] == True) &
    ((prediction_data['Prediction_Numeric'] >= sell_threshold) | 
     (prediction_data['Prediction_Numeric'].isna()))
)
count_max_without_low_pred = max_without_low_prediction.sum()

# Count total low predictions and total maxima for reference
total_low_predictions = (prediction_data['Prediction_Numeric'] < sell_threshold).sum()
total_maxima = prediction_data['Is_Local_Max'].sum()

# Count successful alignments (both sell conditions met)
sell_accurate_count = len(sell_accurate)

print(f"Total points with prediction < {sell_threshold}: {total_low_predictions}")
print(f"Total local maxima points: {total_maxima}")
print(f"Points where prediction < {sell_threshold} but NOT at local maximum: {count_low_pred_not_max}")
print(f"Local maxima points without prediction < {sell_threshold}: {count_max_without_low_pred}")
print(f"Successful alignments (prediction < {sell_threshold} AT local maximum): {sell_accurate_count}")

if total_low_predictions > 0:
    sell_precision = sell_accurate_count / total_low_predictions * 100
    print(f"Sell Precision: {sell_precision:.2f}% of low predictions are at local maxima")

if total_maxima > 0:
    sell_recall = sell_accurate_count / total_maxima * 100
    print(f"Sell Recall: {sell_recall:.2f}% of local maxima have low predictions")

# Calculate overall accuracy metrics
print("\n3. OVERALL ALIGNMENT METRICS:")

total_extrema = total_minima + total_maxima
total_aligned = accurate_count + sell_accurate_count
total_predictions = len(prediction_data)
total_predictions_with_values = prediction_data['Prediction_Numeric'].notna().sum()

if total_extrema > 0:
    extrema_alignment_rate = total_aligned / total_extrema * 100
    print(f"Overall extrema alignment rate: {extrema_alignment_rate:.2f}% of all extrema points have correctly aligned predictions")

if total_predictions_with_values > 0:
    prediction_accuracy = total_aligned / total_predictions_with_values * 100
    print(f"Overall prediction accuracy: {prediction_accuracy:.2f}% of all predictions correctly identify extrema points")

accuracy_value = (len(accurate)+len(sell_accurate))/(len(accurate)+len(sell_accurate)+count_high_pred_not_min+count_low_pred_not_max)
print(f"Overall accuracy value: {accuracy_value}")

try:
    # Ensure openpyxl is installed: pip install openpyxl
    print(f"\nAttempting to save DataFrame (including index) to: {excel_filepath}")

    # **** CHANGE: Removed index=False ****
    prediction_data.to_excel(
        excel_writer=excel_filepath,
        sheet_name='Predictions'
        # index=True is the default, so removing index=False works
    )

    print(f"DataFrame successfully saved to {excel_filepath}")

except ImportError:
    print("\nError saving to Excel: The 'openpyxl' library is required.")
    print("Please install it using the command:")
    print("pip install openpyxl")
except Exception as e:
    print(f"\nAn error occurred while saving the DataFrame to Excel: {e}")