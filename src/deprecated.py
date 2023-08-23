### This file contains code that is not used anymore but could be used in the future


# Manual training 
epochs = 5
batch_size = 128

for epoch in range(epochs):
    
    train_x = next(train_generator)  # Assuming the generator yields (batch_x, label_y)
    validation_x = next(validation_generator)  # Assuming the generator yields (batch_x, label_y)
        
    wathes = train_x # np.reshape(train_x, (len(train_x),NUM_PIXELS))
    watches_val =  validation_x #np.reshape(validation_x, (len(validation_x),NUM_PIXELS))

    history = vae.fit(
        wathes, 
        y=wathes, 
        validation_data=(watches_val, watches_val), 
        batch_size=batch_size, 
        verbose=0)
   
    loss = history.history["loss"][0]
    val_loss = history.history["val_loss"][0]
    loss_metrics.append(loss)
    val_metrics.append(val_loss)
    print(f"Epoch {epoch+1}/{epochs} | loss: {loss} | val: {val_loss}")
print("end")

train_x = next(validation_generator)
wathes = train_x[0][:5]
print(wathes.shape)
ploters.plot_generated_images([wathes], 1, 5)
wathes = np.reshape(wathes, (len(wathes), NUM_PIXELS))
result = vae.predict(wathes)
result = np.reshape(result, (len(result), 80, 60, 1))
ploters.plot_generated_images([result], 1, 5)

history.history["loss"] = loss_metrics
history.history["val_loss"] = val_metrics
ploters.plot_history(history)