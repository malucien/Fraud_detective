import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
        # Import your fraud detection model code
from fraud_detection_model import predict_fraud

# Set up OpenAI API credentials
openai.api_key = "YOUR_API_KEY"

# Load the model
clf = RandomForestClassifier(n_estimators=100, max_depth=10)

# Load the dataset and train the model
df = pd.read_csv('credit_card_data.csv')
df['transaction_hour'] = pd.to_datetime(df['transaction_time']).dt.hour
df['merchant_category'] = df['merchant'].apply(lambda x: x.split(':')[0])
df['is_high_amount'] = df['transaction_amount'] > 1000
df['is_frequent_user'] = df.groupby('user_id')['user_id'].transform('count') > 10
df['transaction_frequency'] = df.groupby('user_id')['user_id'].transform('count')
ip_data = pd.read_csv('ip_address_data.csv')
df = df.merge(ip_data, on='ip_address', how='left')
df['is_high_risk_country'] = df['country'].apply(lambda x: x in ['Nigeria', 'Indonesia', 'Brazil'])
X = df[['user_age', 'transaction_amount', 'transaction_hour', 'merchant_category', 'is_high_amount', 'is_frequent_user', 'transaction_frequency', 'is_high_risk_country']]
y = df['is_fraud']
clf.fit(X, y)

# Define a function to process a new transaction
def process_transaction(transaction_data):
    # transaction_data should be a dictionary containing the transaction details
    # For example:
    # transaction_data = {
    #     'user_id': '12345',
    #     'transaction_amount': 100.0,
    #     'merchant': 'Amazon:Books',
    #     'transaction_time': '2022-04-01 12:34:56',
    #     'ip_address': '192.168.1.1',
    #     ...
    # }

    # Create a new DataFrame with the transaction data
    transaction_df = pd.DataFrame(transaction_data, index=[0])
    transaction_df['transaction_hour'] = pd.to_datetime(transaction_df['transaction_time']).dt.hour
    transaction_df['merchant_category'] = transaction_df['merchant'].apply(lambda x: x.split(':')[0])
    transaction_df['is_high_amount'] = transaction_df['transaction_amount'] > 1000
    transaction_df['is_frequent_user'] = df.groupby('user_id')['user_id'].transform('count') > 10
    transaction_df['transaction_frequency'] = df.groupby('user_id')['user_id'].transform('count')
    transaction_df = transaction_df.merge(ip_data, on='ip_address', how='left')
    transaction_df['is_high_risk_country'] = transaction_df['country'].apply(lambda x: x in ['Nigeria', 'Indonesia', 'Brazil'])
    X_new = transaction_df[['user_age', 'transaction_amount', 'transaction_hour', 'merchant_category', 'is_high_amount', 'is_frequent_user', 'transaction_frequency', 'is_high_risk_country']]

    # Make a prediction on the new transaction
    y_pred = clf.predict(X_new)
    y_prob = clf.predict_proba(X_new)

    # Analyze the prediction
    if y_pred[0] == 1 and y_prob[0][1] > 0.8:
        # The model predicts that the transaction is fraudulent
        # Take action based on the prediction (e.g. block the transaction and alert a human analyst)
        print('Fraudulent transaction detected:', transaction_data)
    else:
        # The model predicts that the transaction is not fraudulent
        # Transaction is processed
        print('Legitimate')



# Define a test dataset
test_data = [
    {'user_id': 123, 'transaction_amount': 500, 'merchant': 'Amazon', 'transaction_time': '2022-03-28 12:30:00', 'ip_address': '123.456.789.0'},
    {'user_id': 456, 'transaction_amount': 1000, 'merchant': 'Walmart', 'transaction_time': '2022-03-28 15:45:00', 'ip_address': '987.654.321.0'},
    # Add more test data here...
]

# Loop through the test data and make predictions using your model
num_correct_predictions = 0
for data in test_data:
    prediction = predict_fraud(data)
    if prediction == data['is_fraud']:
        num_correct_predictions += 1

# Calculate the accuracy of your model on the test data
accuracy = num_correct_predictions / len(test_data)
print('Accuracy:', accuracy)

