import streamlit as st
import torch
import json
import pandas as pd


def chat_interface(model, tokenizer, num_labels, label_encoders):
    st.title("LAM Task Classifier")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Recreate the assistant's response
                # Check if content is a string (old messages)
                if isinstance(message["content"], str):
                    # Try to parse it as JSON
                    try:
                        response_content = json.loads(message["content"])
                    except json.JSONDecodeError:
                        # If parsing fails, display the content as is
                        st.markdown(message["content"])
                        continue
                else:
                    response_content = message["content"]

                st.markdown(response_content['user_friendly_explanation'])

                # Display the predictions in a modern layout
                display_predictions(response_content['predictions'])

                # Display the custom message
                st.markdown("---")  # Separator
                if response_content['message_type'] == 'error':
                    st.error(f"**Decision:** {response_content['custom_message']}")
                elif response_content['message_type'] == 'warning':
                    st.warning(f"**Decision:** {response_content['custom_message']}")
                elif response_content['message_type'] == 'success':
                    st.success(f"**Decision:** {response_content['custom_message']}")
                else:
                    st.info(f"**Decision:** {response_content['custom_message']}")
            else:
                # For user messages, just display the content
                st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Enter a task description:"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get the model's prediction
        with torch.no_grad():
            logits = model(**inputs)

        # Map model outputs to user-friendly task labels
        task_labels = {
            'r_l': 'Risk level of the task',
            'f': 'Feasibility of the task',
            'l': 'Legality of the task',
            'e': 'Ethicality of the task',
            'rv': 'Reversibility of the task',
            'li': 'Limitation of the task'
        }

        # Convert logits to predictions
        predictions = {}
        for key in logits:
            _, preds = torch.max(logits[key], dim=1)
            # Fix the label decoding (reverse if necessary)
            label_order = label_encoders[key]
            if key == 'r_l':  # Invert if needed
                label_order = label_order[::-1]
            predicted_label = label_order[preds.item()]
            predictions[task_labels[key]] = predicted_label  # Human-readable label with new task labels

        # Extract relevant labels for decision logic
        risk_level = predictions.get('Risk level of the task')
        legality = predictions.get('Legality of the task')
        ethicality = predictions.get('Ethicality of the task')

        # Determine the custom message based on predictions
        if risk_level == 'High' and legality == 'No' and ethicality == 'No':
            message = "This task is not legal and it's risky to do."
            message_type = 'error'
        elif risk_level == 'Low' and legality == 'No':
            message = "It won't proceed because it's not legal."
            message_type = 'error'
        elif risk_level == 'Low' and legality == 'Yes' and ethicality == 'Yes':
            message = "Task is perfectly fine to execute."
            message_type = 'success'
        elif risk_level == 'High' and legality == 'Yes' and ethicality == 'Yes':
            message = "The task is risky to do; please consider the consequences."
            message_type = 'warning'
        else:
            message = "Please review the predicted labels for a detailed analysis."
            message_type = 'info'

        # Prepare the response
        response_content = {
            "predictions": predictions,
            "user_friendly_explanation": "Here are the predicted task labels based on your input:",
            "custom_message": message,
            "message_type": message_type
        }

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response_content['user_friendly_explanation'])

            # Display the predictions in a modern layout
            display_predictions(response_content['predictions'])

            # Display the custom message
            st.markdown("---")  # Separator
            if response_content['message_type'] == 'error':
                st.error(f"**Decision:** {response_content['custom_message']}")
            elif response_content['message_type'] == 'warning':
                st.warning(f"**Decision:** {response_content['custom_message']}")
            elif response_content['message_type'] == 'success':
                st.success(f"**Decision:** {response_content['custom_message']}")
            else:
                st.info(f"**Decision:** {response_content['custom_message']}")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_content})

    # Add a sidebar with additional information or controls if needed
    with st.sidebar:
        st.header("About")
        st.markdown("This is a LAM Task Classifier that helps categorize tasks based on various features.")

        # Add a button to clear the chat history
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()

def get_color(label, value):
    # Function to map label and value to a specific color
    if label == 'Risk level of the task':
        if value == 'High':
            return '#FF4B4B'  # Red
        elif value == 'Medium':
            return '#FF914D'  # Dark Orange
        elif value == 'Low':
            return '#4CAF50'  # Green
    elif label in ['Feasibility of the task', 'Legality of the task', 'Ethicality of the task', 'Reversibility of the task']:
        if value == 'Yes':
            return '#4CAF50'  # Green
        elif value == 'No':
            return '#FF4B4B'  # Red
    elif label == 'Limitation of the task':
        if value == 'Yes':
            return '#FF4B4B'  # Red
        elif value == 'No':
            return '#4CAF50'  # Green
    # Default color
    return '#D3D3D3'  # Light Gray

def display_predictions(predictions):
    # Function to display predictions in a modern layout using columns
    labels = list(predictions.keys())
    values = list(predictions.values())

    # Calculate the number of columns per row
    cols_per_row = 2  # Adjusted for better spacing
    total_labels = len(labels)
    rows = (total_labels + cols_per_row - 1) // cols_per_row  # Ceiling division

    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row, gap="large")
        for col in cols:
            if idx < total_labels:
                label = labels[idx]
                value = values[idx]
                color = get_color(label, value)
                text_color = 'white' if color != '#FF914D' else 'black'  # Ensure text is readable
                with col:
                    # Create a styled box using HTML
                    html_content = f'''
                    <div style="
                        background-color: {color};
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        color: {text_color};
                        font-weight: bold;
                        margin-bottom: 10px;
                    ">
                        <h4 style="margin: 0 0 10px 0;">{label}</h4>
                        <p style="font-size:24px; margin: 0;">{value}</p>
                    </div>
                    '''
                    st.markdown(html_content, unsafe_allow_html=True)
                idx += 1
            else:
                break
