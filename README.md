

# Neural Network Chatbot (Testing Project)

This is a basic chatbot project created using a neural network to test and showcase skill development. The data used in this project is minimal, so the chatbot might not be fully accurate, but it works quite well given the scope.

## Files Provided:
- `chatbot.py`: This file creates and trains the chatbot model.
- `deploy.py`: This file uses the trained model to interact with users.
- `appointments.csv`: A sample file containing appointment data for demonstration purposes.
- `classes.pkl`, `words.pkl`: These files store the class and word data used by the model.
- `model.h5`: The trained neural network model.
- `data.json`: The dataset used for training the chatbot, containing intents and responses.

## How to Run:
1. **Train the Model:**  
   First, run the `chatbot.py` file to build and train the model. This script handles the neural network training using the provided dataset (`data.json`).
   
   ```bash
   python chatbot.py
   ```

   The model will be saved as `model.h5`, and relevant class and word data will be saved as `classes.pkl` and `words.pkl`.

2. **Deploy the Chatbot:**  
   After training, run the `deploy.py` file to interact with the chatbot using the trained model.

   ```bash
   python deploy.py
   ```

   The chatbot will use the `model.h5`, `classes.pkl`, and `words.pkl` files to generate responses.

## Understanding the Code:
- The Python code contains comments to explain the functions and key steps in the process, making it easier to follow and understand.
- Feel free to explore and tweak the code to experiment with different datasets or neural network configurations.

## Notes:
- This project is primarily for testing purposes and does not include a large dataset, so the chatbot's responses might not be highly accurate or sophisticated.
- You can modify the data and improve the model for better results.

