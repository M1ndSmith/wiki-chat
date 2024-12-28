# app.py
from flask import Flask, request, jsonify, render_template
from langchain_groq import ChatGroq
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ChatbotManager:
    def __init__(self):
        self.memory = MemorySaver()
        self.model = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.7,  # Slightly increased for more natural responses
            max_tokens=8192,
            timeout=30,  # Added timeout
        )
        
        # Enhanced prompt template with better context
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a knowledgeable assistant with access to Wikipedia information.
                - Provide accurate, concise responses
                - Always cite your Wikipedia sources
                - If unsure, acknowledge uncertainty
                - Break down complex topics into understandable explanations
                Available tools: {tools}"""
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        self.retriever = WikipediaRetriever(
            top_k_results=3,  # Increased for better context
            load_max_docs=5,
        )
        self.tools = [self.retriever.run]
        self.agent_executor = create_react_agent(
            self.model, 
            self.tools, 
            checkpointer=self.memory
        )
        
        # Conversation history storage
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}

    def get_or_create_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        return self.conversations[session_id]

    async def process_message(self, message: str, session_id: str) -> List[Dict[str, Any]]:
        try:
            conversation = self.get_or_create_conversation(session_id)
            conversation.append({"role": "user", "content": message})
            
            config = {"configurable": {"thread_id": session_id}}
            messages = [HumanMessage(content=message)]
            
            chunks = []
            async for chunk in self.agent_executor.astream(
                {"messages": messages}, 
                config
            ):
                chunks.append(self._sanitize_chunk(chunk))
            
            # Extract final response
            final_response = self._extract_final_response(chunks)
            if final_response:
                conversation.append({"role": "assistant", "content": final_response})
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise

    def _sanitize_chunk(self, obj: Any) -> Any:
        """Sanitize chunks for JSON serialization with improved type handling"""
        if isinstance(obj, dict):
            return {k: self._sanitize_chunk(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_chunk(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif hasattr(obj, "content"):
            return {"content": obj.content}
        else:
            return str(obj)

    def _extract_final_response(self, chunks: List[Dict[str, Any]]) -> str:
        """Extract the final response from chunks with error handling"""
        try:
            for chunk in reversed(chunks):
                if chunk.get("agent", {}).get("messages"):
                    messages = chunk["agent"]["messages"]
                    if messages and isinstance(messages[-1], dict):
                        return messages[-1].get("content", "")
            return "No response generated."
        except Exception as e:
            logger.error(f"Error extracting final response: {e}")
            return "Error processing response."

# Initialize chatbot manager
chatbot = ChatbotManager()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
async def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        session_id = request.headers.get("X-Session-ID", "default")
        
        if not user_input:
            return jsonify({"error": "Empty message"}), 400
            
        chunks = await chatbot.process_message(user_input, session_id)
        return jsonify(chunks), 200
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True,threaded=True)