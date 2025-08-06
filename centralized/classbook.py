import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

def load_and_preprocess_data(file_path):
    """Load dataset and preprocess it by mapping attack types and selecting features."""
    df = pd.read_csv(file_path, low_memory=False)
    # df.drop(columns=['Unnamed: 0'], inplace=True)
    # df = df.iloc[:, 2:].reset_index(drop=True)
    
    # Mapping attack types to numerical labels
    attacks = {'Normal': 0, 'MITM': 1, 'Uploading': 2, 'Ransomware': 3, 'SQL_injection': 4,
               'DDoS_HTTP': 5, 'DDoS_TCP': 6, 'Password': 7, 'Port_Scanning': 8,
               'Vulnerability_scanner': 9, 'Backdoor': 10, 'XSS': 11, 'Fingerprinting': 12,
               'DDoS_UDP': 13, 'DDoS_ICMP': 14}
    df['Attack_type'] = df['Attack_type'].map(attacks)
    
    X = df.drop(columns=['Attack_label', 'Attack_type'])
    y = df['Attack_label']
    
    return X, y

def feature_selection(X, y):
    """Select best features using Chi-Squared test."""
    # Ensure non-negative data for chi2
    X_non_negative = X.where(X >= 0, 0)
    chi_selector = SelectKBest(chi2, k='all')
    chi_selector.fit_transform(X_non_negative, y)
    
    chi_scores = pd.DataFrame({'feature': X.columns, 'score': chi_selector.scores_}).dropna()
    chi_scores = chi_scores.sort_values(by='score', ascending=False)
    selected_features = chi_scores['feature'].tolist()
    
    return selected_features

def prepare_data(X, y, selected_features, n_components=30):
    """Scale, apply PCA, and split data into training, validation, and test sets."""
    # Select features
    X_selected = X[selected_features]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Generate scree plot
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--', label='Explained Variance Ratio')
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', label='Cumulative Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot for PCA')
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/PCA/scree_plot.jpg')
    plt.close()
    
    # Print PCA results
    print("Explained Variance Ratios:", explained_variance_ratio)
    print("Cumulative Explained Variance:", cumulative_variance)
    print(f"Number of components retained: {n_components}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, pca

def cnn_lstm_gru_model(input_shape):
    """Define and compile CNN-LSTM-GRU model."""
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=True),
        GRU(64, return_sequences=False),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train the model and evaluate its performance."""
    start_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=6, batch_size=32)
    train_time = time.time() - start_time
    
    start_time = time.time()
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
    test_time = time.time() - start_time
    
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Training time: {train_time:.2f} seconds')
    print(f'Testing time: {test_time:.2f} seconds')
    
    return history, model

def plot_metrics(history, state):
    """Plot training and validation accuracy and loss."""
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'../results/PCA/{state}/accuracy_plot.jpg')
    plt.close()
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'../results/PCA/{state}/loss_plot.jpg')
    plt.close()

def evaluate_model(model, X_test, y_test, state):
    """Evaluate the model and generate a confusion matrix."""
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    print(classification_report(y_test, y_pred_binary, target_names=['No Intrusion', 'Intrusion']))
    conf_mat = confusion_matrix(y_test, y_pred_binary)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['No Intrusion', 'Intrusion'], yticklabels=['No Intrusion', 'Intrusion'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(f'../results/PCA/{state}/confusion_matrix.jpg')
    plt.close()
    
    # Plot normalized confusion matrix
    cm_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, cmap='Blues', xticklabels=['No Intrusion', 'Intrusion'], yticklabels=['No Intrusion', 'Intrusion'], fmt='.2%')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Normalized Confusion Matrix (Percentages)')
    plt.savefig(f'../results/PCA/{state}/normalized_confusion_matrix.jpg')
    plt.close()

def load_and_preprocess_test_data(file_path, intended_columns, selected_features, scaler, pca):
    """Load and preprocess test data, applying the same scaling and PCA transformation."""
    df = pd.read_csv(file_path)
    for column in intended_columns:
        if column not in df.columns:
            df[column] = 0  # Default value for missing columns
    
    # Drop columns not in intended_columns
    list_a = df.columns.to_list()
    list_b = intended_columns
    result = [item for item in list_a if item not in list_b]
    df.drop(columns=result, inplace=True)
    
    # Select features and apply scaling and PCA
    df = df[selected_features]
    X_test_scaled = scaler.transform(df)
    X_test_pca = pca.transform(X_test_scaled)
    
    return X_test_pca

def main():
    """Main execution function."""
    file_path = '/Users/neupanek/Downloads/FL-IDS/federated/federated_datasets/train_data.csv'
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path)
    
    # Perform feature selection
    selected_features = feature_selection(X, y)
    
    # Prepare data with PCA
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, pca = prepare_data(X, y, selected_features, n_components=20)
    
    # Reshape data for CNN input (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Define and train model
    input_shape = (X_train.shape[1], 1)
    model = cnn_lstm_gru_model(input_shape)
    model.summary()
    plot_model(model, to_file='../results/PCA/model_structure.png', show_shapes=True)
    
    history, model = train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test)
    plot_metrics(history, 'train')
    evaluate_model(model, X_test, y_test, 'train')
    
    model.save('..../results/PCA/cnn_lstm_gru_model_binary_pca.h5')
    
    # Test with attack datasets
    test_file_path = '/Users/neupanek/Downloads/FL-IDS/federated/federated_datasets/test_data.csv'
    df_before = pd.read_csv(test_file_path)
    X_test_pca = load_and_preprocess_test_data(test_file_path, selected_features, selected_features, scaler, pca)
    X_test_pca = X_test_pca.reshape(X_test_pca.shape[0], X_test_pca.shape[1], 1)
    y_test = df_before['Attack_label']
    evaluate_model(model, X_test_pca, y_test, 'test')

if __name__ == "__main__":
    main()