# Prepare the dataset
train_generator = ...
val_generator = ...
test_generator = ...

# Compile the model
model = unet(input_shape=(256, 256, 3), num_classes=1, filters=[64, 128, 256, 512, 1024], kernel_size=3, activation='relu', 
             padding='same', kernel_initializer='he_normal')
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(lr=0.001), 
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.F1Score()])

# Train the model
history = model.fit(train_generator, epochs=100, batch_size=32, validation_data=val_generator, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

# Evaluate the model on the testing set
test_metrics = model.evaluate(test_generator)
print(f'Test loss: {test_metrics[0]}, Test accuracy: {test_metrics[1]}, Test precision: {test_metrics[2]}, Test recall: {test_metrics[3]}, Test F1-score: {test_metrics[4]}')