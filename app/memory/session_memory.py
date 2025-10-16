from langchain.memory import ConversationBufferMemory

session_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)