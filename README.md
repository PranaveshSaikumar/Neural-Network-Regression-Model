# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective is to create a neural network regression model that learns the relationship between input features and continuous target variables, enabling accurate predictions on unseen data from a given dataset.

## Neural Network Model

![image](https://github.com/user-attachments/assets/4aa1ed46-43a1-46dd-886f-06d0a9bb8340)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Pranavesh Saikumar
### Register Number: 212223040149
```
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 12)
        self.fc3 = nn.Linear(12, 1)
        self.relu = nn.ReLU()
        self.history = {'loss':[]}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

```
<br><br><br><br><br><br><br><br>
## Dataset Information

![image](https://github.com/user-attachments/assets/55dce5a4-b4f9-485f-a362-d2721bd34448)
<br><br><br><br><br><br><br><br><br><br><br><br><br><br>
## OUTPUT
### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/658fe927-6f18-4415-b6cd-07eb443e3f4c)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/3eb0cb22-9908-4b42-8eee-553ec29d89c7)

## RESULT

Thus, a neural network regression model for the given dataset is created successfully.
