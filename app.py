import streamlit as st

from RAG.tools.response_generator import generate_with_rag


def main():
    st.set_page_config(page_title="Conversational Interface", page_icon="logo.png")
    st.title('RAG Generation')

    # Load the logo image
    st.image("logo.png", width=1000)

    st.title("Conversational Interface")

    # Initialize the conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Display the conversation history
    for message in st.session_state.conversation_history:
        if message["sender"] == "user":
            st.write(f"User: {message['text']}")
        else:
            st.write(f"Assistant: {message['text']}")

    # Get user input
    user_input = st.text_input("Enter your query:", "")

    # Generate a response using the backend function
    if user_input and st.button("Submit"):
        response = generate_with_rag(user_input)
        st.session_state.conversation_history.append({"sender": "user", "text": user_input})
        st.session_state.conversation_history.append({"sender": "assistant", "text": response})
        st.experimental_rerun()


if __name__ == "__main__":
    main()