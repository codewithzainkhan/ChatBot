import csv
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import pickle
import random
import re

# Initialize lemmatizer and load model
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')

# Load intents and words
intents = json.loads(open('data.json').read())

with open('words.pkl', 'rb') as file:
    words = pickle.load(file)

with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

# Dictionary to keep track of user data
user_data = {
    'name': None,
    'contact': None,
    'appointment_date': None,
    'appointment_time': None
}

# Load available dates and times from CSV
def load_appointments():
    appointments = {}
    with open('appointments.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            date = row['date']
            time = row['time']
            status = row['status']
            if date not in appointments:
                appointments[date] = {}
            appointments[date][time] = status
    return appointments

def save_appointments(appointments):
    with open('appointments.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['date', 'time', 'status'])
        for date, times in appointments.items():
            for time, status in times.items():
                writer.writerow([date, time, status])

appointments = load_appointments()

# Text preprocessing functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def extract_user_data(text, user_data,tag):
    # Example regex patterns to extract data
 if tag == 'provide_name':
    name_match = re.search(r'my name is (\w+)', text, re.IGNORECASE)
    if name_match:
        user_data['name'] = name_match.group(1)
 elif tag == 'provide_contact':
    contact_match = re.search(r'contact is (\S+)', text, re.IGNORECASE)
    if contact_match:
        user_data['contact'] = contact_match.group(1)
 elif not user_data['appointment_date']:
    date_match = re.search(r'(?:on|for) (\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
    if date_match:
        user_data['appointment_date'] = date_match.group(1)
 elif not user_data['appointment_time']:
    time_match = re.search(r'at (\d{1,2}:\d{2}\s*(?:AM|PM)?)', text, re.IGNORECASE)
    if time_match:
        user_data['appointment_time'] = time_match.group(1).strip().upper()
        print(f"Extracted appointment time: {user_data['appointment_time']}")
    else:
        print("No valid time found in the text.")

def get_response(ints, intents_json, user_data, user_message):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    response = "Sorry, I didn't understand that."

    # Extract user data from the actual user message
    extract_user_data(user_message, user_data,tag)
    print(f"User data after extraction: {user_data}")  # Debug print

    for i in list_of_intents:
        if i['tag'] == tag:
            response = random.choice(i['responses'])
            break

    # Handle custom responses for specific intents
    if tag == 'book_appointment':
        if not user_data['name']:
            response = "Sure, I can help with booking an appointment. Please provide your name."
        elif not user_data['contact']:
            response = f"Got it, {user_data['name']}! Please provide your contact information."
        elif not user_data['appointment_date']:
            response = f"Great! Please choose a date from the following available dates: {', '.join(available_dates())}."
        elif not user_data['appointment_time']:
            response = f"Thank you! Now, please choose a time from the following available times: {', '.join(available_times(user_data['appointment_date']))}."
        else:
            date = user_data['appointment_date']
            time = user_data['appointment_time']
            if appointments.get(date) and appointments[date].get(time) == 'available':
                response = f"Your appointment has been booked for {date} at {time}. We will contact you at {user_data['contact']}. Thank you!"
                appointments[date][time] = 'booked'
                save_appointments(appointments)
                user_data['name'] = None
                user_data['contact'] = None
                user_data['appointment_date'] = None
                user_data['appointment_time'] = None
            else:
                response = "The selected time is no longer available. Please choose a different time."

    elif tag == 'provide_name':
        if not user_data['name']:
            response = "Please provide your name."
        elif not user_data['contact']:
            response = f"Nice to meet you, {user_data['name']}! Can you please provide your contact information?"

    elif tag == 'provide_contact':
        if not user_data['contact']:
            response = "Please provide your contact information."
        else:
            response = f"Thank you, {user_data['name']}! Your contact information has been noted. Let's proceed with booking an appointment. Please choose a date from the following available dates: {', '.join(available_dates())}."

    elif tag == 'provide_date':
        if user_data['appointment_date']:
            response = f"Date {user_data['appointment_date']} noted! Please choose a time from the following available times: {', '.join(available_times(user_data['appointment_date']))}."


    elif tag == 'provide_time':
        if user_data['appointment_date']:
            if appointments.get(user_data['appointment_date']):
                if appointments[user_data['appointment_date']].get(user_data['appointment_time']) == 'available':
                    response = f"Time {user_data['appointment_time']} noted! Your appointment has been booked. We will contact you at {user_data['contact']}."
                    appointments[user_data['appointment_date']][user_data['appointment_time']] = 'booked'
                    save_appointments(appointments)
                    user_data['name'] = None
                    user_data['contact'] = None
                    user_data['appointment_date'] = None
                    user_data['appointment_time'] = None
                else:
                    response = "The selected time is no longer available. Please choose a different time."
            else:
                response = "The selected date is not available. Please choose a different date."


    return response





def available_dates():
    return [date for date in appointments if any(status == 'available' for status in appointments[date].values())]

def available_times(date):
    if date in appointments:
        return [time for time, status in appointments[date].items() if status == 'available']
    return []

def chatbot_response(text, user_data):
    ints = predict_class(text, model)
    res = get_response(ints, intents, user_data, text)
    return res


# Test the chatbot
print("Welcome to the chatbot! Type 'quit' to exit.")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    message = message.lower()
    response = chatbot_response(message, user_data)
    print("Bot:", response)
