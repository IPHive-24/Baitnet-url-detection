
    import numpy as np
    import pandas as pd
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.metrics import confusion_matrix, classification_report
    from imblearn.over_sampling import SMOTE
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import label_data

    # Load the dataset
    urls = label_data.main()

    samples = []
    labels = []
    for k, v in urls.items():
        samples.append(k)
        labels.append(v)

    print(labels.count(1))
    print(labels.count(0))

    # Tokenization for URLs (Character-level)
    max_chars = 20000
    maxlen = 128

    tokenizer = Tokenizer(num_words=max_chars, char_level=True)
    tokenizer.fit_on_texts(samples)
    sequences = tokenizer.texts_to_sequences(samples)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Convert sequences to character-level padded sequences
    X_padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post', value=0)

    # Convert data to NumPy arrays
    X_padded = np.array(X_padded)
    labels = np.asarray(labels)
    print('Shape of data tensor:', X_padded.shape)
    print('Shape of label tensor:', labels.shape)


    # Split into training and testing sets
    training_samples = int(len(samples) * 0.50)
    validation_samples = int(len(labels) * 0.50)
    print(training_samples, validation_samples)

    indices = np.arange(X_padded.shape[0])
    np.random.shuffle(indices)
    data = X_padded[indices]
    labels = labels[indices]
    x = data[:training_samples]
    y = labels[:training_samples]
    x_test = data[training_samples: training_samples + validation_samples]
    y_test = labels[training_samples: training_samples + validation_samples]


    # Perform oversampling of the minority class using SMOTE
    smote = (sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(x, y)

    embedding_dim = 300

    # Build a simplified CNN model with Leaky ReLU and BatchNormalization
    model = Sequential()
    model.add(Embedding(max_chars, embedding_dim, input_length=maxlen))
    model.add(Conv1D(filters=64, kernel_size=3))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Define ModelCheckpoint callback to save the best model
    checkpoint = ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1,
        save_weights_only=False
    )

    # Train the model on the resampled data with callbacks
    batch_size = 20000  
    # Fit the model                 
    epochs = 5
    history = model.fit(X_train_resampled, y_train_resampled, batch_size=batch_size, epochs=epochs, 
                        validation_split=0.20, shuffle=True, 
                        validation_data=(x_test, y_test), 
                        callbacks=[early_stopping, checkpoint])

    # Evaluate the model
    # Print training accuracy and training loss for each epoch
    for epoch, hist in enumerate(history.history['accuracy']):
        print(f"Epoch {epoch + 6}: Training Accuracy = {hist:.4f}, Training Loss = {history.history['loss'][epoch]:.4f}")

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Confusion matrix and classification report
    y_pred = model.predict(x_test)
    y_pred_binary = np.round(y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary, target_names=['Legitimate', 'Phishing']))

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_binary)
    print("Accuracy:", accuracy)
    # Calculate precision
    precision = precision_score(y_test, y_pred_binary)
    print("Precision:", precision)
    # Calculate recall
    recall = recall_score(y_test, y_pred_binary)
    print("Recall:", recall)
    # Calculate F1-score
    f1 = f1_score(y_test, y_pred_binary)
    print("F1 Score:", f1)
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC Score:", roc_auc)
    # Save the final model
    model.save("1.h5")
    print("Model saved successfully.")