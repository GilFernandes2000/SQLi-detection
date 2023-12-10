import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Function to load data from a list of text files
def load_data(file_paths, label):
    data_list = []
    for file_path in file_paths:
        data = pd.read_csv(file_path, header=None, sep=" ")  # Adjust sep if the separator is different
        data['label'] = label  # Add a column for the label
        data_list.append(data)
    return pd.concat(data_list, ignore_index=True)

# File paths
normal_files = ['features/normal1_features.dat', 'features/normal2_features.dat', 'features/normal3_features.dat']  # Replace with your actual file paths
malicious_file = ['features/blindboolean_features.dat']  # Replace with your actual file path

# Load and concatenate data
normal_data = load_data(normal_files, 0)  # Label normal data as 0
malicious_data = load_data(malicious_file, 1)  # Label malicious data as 1
all_data = pd.concat([normal_data, malicious_data], ignore_index=True)

# Shuffle the data
all_data = all_data.sample(frac=1).reset_index(drop=True)

# Split features and labels
X = all_data.iloc[:, :-1].values
y = all_data.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=28)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Neural network architecture
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.summary()
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['accuracy'])

# Train the model
losses = model.fit(X_train, y_train, epochs=16, batch_size=128, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Make predictions on the test set
predictions = model.predict(X_test)
predicted_labels = ['Malicious' if pred > 0.5 else 'Normal' for pred in predictions]

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

for i in range(len(X_test)):  # Adjust the range as needed
    print(f"Test Sample {i+1}: Predicted - {predicted_labels[i]}, Actual - {'Malicious' if y_test[i] else 'Normal'}")

    if predicted_labels[i] == 'Malicious' and y_test[i] == 1:
        true_positives += 1
    elif predicted_labels[i] == 'Malicious' and y_test[i] == 0:
        false_positives += 1
    elif predicted_labels[i] == 'Normal' and y_test[i] == 1:
        false_negatives += 1
    elif predicted_labels[i] == 'Normal' and y_test[i] == 0:
        true_negatives += 1

print(f"True Positives: {((true_positives/(true_positives+false_positives))*100) if true_positives+false_positives > 0 else 0:.2f}%")
print(f"True Negatives: {((true_negatives/(true_negatives+false_negatives))*100) if true_negatives+false_negatives > 0 else 0:.2f}%")
print(f"False Positives: {((false_positives/(true_positives+false_positives))*100) if (true_positives+false_positives) > 0 else 0:.2f}%")
print(f"False Negatives: {((false_negatives/(true_negatives+false_negatives))*100) if (true_negatives+false_negatives) > 0 else 0:.2f}%")

loss_df = pd.DataFrame(losses.history)
plt.plot(loss_df.loc[:,['loss','accuracy']])
plt.title('Training Loss and Accuracy')
plt.legend(['Loss', 'Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.show()
