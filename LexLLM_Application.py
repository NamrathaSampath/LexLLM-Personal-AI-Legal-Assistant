import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from cassio import init
from descope.descope_client import DescopeClient
from descope.exceptions import AuthException
import PyPDF2
import bcrypt
import uuid
import openai
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Access environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

DESCOPE_PROJECT_ID = str(st.secrets.get("DESCOPE_PROJECT_ID"))
descope_client = DescopeClient(project_id=DESCOPE_PROJECT_ID)

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize Astra DB connection
init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Set up Streamlit UI configurations
st.set_page_config(page_title="LexLLM", page_icon="📄", layout="centered")

# Default admin credentials
DEFAULT_USERNAME = "admin123"
DEFAULT_PASSWORD = "Admin123"

# Constants for vector store collections
PERSISTENT_COLLECTION = "persistent_docs_collection"

def init_session_state():
    """Initialize session state variables"""
    states = {
        "logged_in": False,
        "chat_history": [],
        "conversation_chain": None,
        "vectorstore": None,
        "uploaded_pdf_text": "",
        "current_model": None,
        "is_doc_mode": False,
        "model_type": "base",
        "experience_level": "Intermediate",
        "last_experience_level": "Intermediate",
        "token": None,
        "refresh_token": None,
        "user": None
    }
    
    for key, value in states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_model_base(model_name):
    """Extract base model name from variant"""
    return model_name.split('-')[0]

def setup_vectorstore(collection_name=PERSISTENT_COLLECTION):
    """Set up vector store with specified collection name"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Cassandra(
        embedding=embeddings,
        table_name=collection_name,
        session=None,
        keyspace=None
    )
    return vectorstore

def get_llm(model_name):
    """Initialize and return the specified language model"""
    try:
        if model_name == "llama":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found")
            if "-ft" in model_name:
                try:
                    # Using OpenAI client for direct API access
                    client = openai.OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
                    # Test with a simple query to ensure model accessibility
                    response = client.chat.completions.create(
                        model="llama-3.1-70b-versatile",
                        temperature=0,
                        messages=[{"role": "system", "content": "You are a legal AI assistant."}]
                    )
                    # If successful, return ChatGroq instance
                    return ChatGroq(
                        model="llama-3.1-70b-versatile",
                        temperature=0,
                        api_key=GROQ_API_KEY
                    )
                except Exception as e:
                    st.warning(f"Fine-tuned model unavailable: {str(e)}. Falling back to base model.")
                    return ChatGroq(
                        model="llama-3.1-70b-versatile",
                        temperature=0,
                        api_key=GROQ_API_KEY
                    )
            else:
                return ChatGroq(
                    model="llama-3.1-70b-versatile",
                    temperature=0,
                    api_key=GROQ_API_KEY
                )
        elif model_name == "gpt":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not found")
            if "-ft" in model_name:
                return ChatOpenAI(
                    model="ft:gpt-4o-2024-08-06:personal:lexllm-test1:AC0Zn95B",
                    temperature=0,
                    api_key=OPENAI_API_KEY
                )
            else:
                return ChatOpenAI(
                    model="gpt-4-turbo-preview",
                    temperature=0,
                    api_key=OPENAI_API_KEY
                )
        elif model_name == "gemini":
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not found")
            if "-ft" in model_name:
                try:
                    genai.configure(api_key=GOOGLE_API_KEY)
                    return ChatGoogleGenerativeAI(
                        model="tunedModels/lexllm-xivv5cqwu2zs",
                        temperature=0,
                        google_api_key=GOOGLE_API_KEY,
                        convert_system_message_to_human=True
                    )
                except Exception as e:
                    st.warning("Fine-tuned model unavailable. Falling back to base Mistral model.")
                    return ChatGoogleGenerativeAI(
                        model="gemini-1.5-pro",
                        temperature=0,
                        google_api_key=GOOGLE_API_KEY,
                        convert_system_message_to_human=True
                    )
            else:
                return ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    temperature=0,
                    google_api_key=GOOGLE_API_KEY,
                    convert_system_message_to_human=True
                )
        elif model_name == "mistral":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found")
            if "-ft" in model_name:
                try:
                    # Using OpenAI client for direct model access
                    client = openai.OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
                    response = client.chat.completions.create(
                        model="ft:open-mistral-7b:c995defc:20241117:3e25cc8e",
                        temperature=0,
                        messages=[{"role": "system", "content": "You are a legal AI assistant."}]
                    )
                    # If successful, return ChatGroq instance
                    return ChatGroq(
                        model="ft:open-mistral-7b:c995defc:20241117:3e25cc8e",
                        temperature=0,
                        api_key=GROQ_API_KEY
                    )
                except Exception as e:
                    st.warning(f"Fine-tuned model unavailable: {str(e)}. Falling back to base model.")
                    return ChatGroq(
                        model="mixtral-8x7b-32768",
                        temperature=0,
                        api_key=GROQ_API_KEY
                    )
            else:
                return ChatGroq(
                    model="mixtral-8x7b-32768",
                    temperature=0,
                    api_key=GROQ_API_KEY
                )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        st.error(f"Error initializing model {model_name}: {str(e)}")
        raise

def get_experience_level_prompt(level):
    """Get prompts with strict experience level differentiation"""
    prompts = {
        "Beginner": """
            STRICT RULES FOR BEGINNER RESPONSE:
            1. NO legal jargon or technical terms whatsoever
            2. If you must use a legal term, follow it with "which simply means [simple explanation]"
            3. Use everyday examples, like comparing legal concepts to common situations
            4. Write as if explaining to someone with no legal knowledge
            5. Use phrases like "in simple terms" and "this basically means"
            6. Break down complex ideas into simple, short sentences
            
            Example style:
            "A contract (which is simply a formal agreement between people) requires..."
            "This is like when you make a promise to your friend..."
            "In simple terms, this means..."
            """,
            
        "Intermediate": """
            STRICT RULES FOR INTERMEDIATE RESPONSE:
            1. Use basic legal terms but always add brief explanations
            2. Include some technical language with context
            3. Balance formal terms with plain language explanation
            4. You may use common Latin phrases but translate them
            5. Reference some laws and regulations with explanations
            6. Use moderate legal terminology
            
            Example style:
            "The doctrine of consideration (meaning something of value must be exchanged)..."
            "Per se (meaning 'by itself')..."
            "Under Section 123, which establishes..."
            """,
            
        "Advanced": """
            STRICT RULES FOR ADVANCED RESPONSE:
            1. Use sophisticated legal terminology freely
            2. Include Latin legal phrases without basic explanations
            3. Cite specific statutes, cases, and precedents
            4. Use technical legal analysis and reasoning
            5. Reference complex legal doctrines and principles
            6. Employ formal legal writing style
            
            Example style:
            "The doctrine of stare decisis compels..."
            "Per the holding in Brown v. Board..."
            "Prima facie evidence suggests..."
            """
    }
    return prompts.get(level, prompts["Intermediate"])

def get_legal_verification_prompt():
    """Get the common legal verification prompt"""
    return """
    Before providing an answer, strictly verify if the question is related to:
    - Law and legal concepts
    - Legal documents and procedures
    - Regulations and compliance
    - Legal rights and obligations
    - Legal system and processes
    
    If the question is NOT related to legal matters:
    1. IMMEDIATELY identify it as non-legal
    2. Respond with: "I am a legal AI assistant. I can only assist with legal-related questions. Your question about [topic] is outside my scope."
    3. DO NOT provide any additional information
    
    For legal questions only, proceed with the analysis as specified below.
    """

def create_base_chain(model_name, experience_level="Intermediate"):
    """Creates a basic conversation chain without retrieval"""
    base_model = get_model_base(model_name)
    llm = get_llm(base_model)
    
    legal_verification = get_legal_verification_prompt()
    experience_prompt = get_experience_level_prompt(experience_level)
    
    template = f"""
        You are a specialized legal AI assistant focused EXCLUSIVELY on legal matters.
        {legal_verification}
        {experience_prompt}
        
        Previous conversation:
        {{history}}
        
        Human: {{input}}
        Assistant: First, strictly verify if this is a legal question. If not, decline. If yes, provide a legal analysis:
    """
    
    base_prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    memory = ConversationBufferMemory(
        input_key="input",
        memory_key="history"
    )
    
    chain = ConversationChain(
        llm=llm,
        prompt=base_prompt,
        memory=memory,
        verbose=True
    )
    return chain

def create_chain_with_retrieval(model_name, vectorstore, experience_level="Intermediate"):
    """Creates a conversation chain with document retrieval"""
    if not vectorstore:
        raise ValueError("Vectorstore is required for RAG models")
        
    base_model = get_model_base(model_name)
    llm = get_llm(base_model)
    
    # Simple retriever setup
    k_docs = 5 if "llama-rag" in model_name else 3
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_docs})
    
    # Basic memory setup
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    # Create the chain with minimal configuration
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return chain

def call_gemini_ft(query, experience_level="Intermediate"):
    """Direct call to Gemini with fallback to base model if fine-tuned is unavailable"""
    experience_prompt = get_experience_level_prompt(experience_level)
    
    if st.session_state["uploaded_pdf_text"]:
        context = f"Here's the context from the uploaded PDF:\n\n{st.session_state['uploaded_pdf_text']}\n\n"
        full_prompt = f"{context}{query}"
    else:
        full_prompt = query

    genai.configure(api_key=GOOGLE_API_KEY)
    try:
        model = genai.GenerativeModel('tunedModels/lexllm-xivv5cqwu2zs')
    except Exception as e:
        st.warning("Fine-tuned model unavailable. Using base Gemini model instead.")
        model = genai.GenerativeModel('gemini-1.5-pro')
    
    system_prompt = f"You are a legal AI assistant. {experience_prompt}"
    
    try:
        response = model.generate_content([
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["I understand. I will act as a legal AI assistant following the specified experience level guidelines."]},
            {"role": "user", "parts": [full_prompt]}
        ])
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        # Fallback to base model with simplified prompt
        try:
            response = model.generate_content(f"{system_prompt}\n\n{full_prompt}")
            return response.text.strip()
        except Exception as e:
            return f"Error: Unable to generate response. Please try another model or contact support. ({str(e)})"

def call_gpt_4o(query, model_id="ft:gpt-4o-2024-08-06:personal:lexllm-test1:AC0Zn95B", experience_level="Intermediate"):
    """Direct call to GPT-4o or fine-tuned model with PDF context if available"""
    experience_prompt = get_experience_level_prompt(experience_level)
    
    if st.session_state["uploaded_pdf_text"]:
        context = f"Here's the context from the uploaded PDF:\n\n{st.session_state['uploaded_pdf_text']}\n\n"
        full_prompt = f"{context}{query}"
    else:
        full_prompt = query

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": f"You are a legal AI assistant. {experience_prompt}"},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def call_mistral_ft(query, experience_level="Intermediate"):
    """Direct call to Mistral fine-tuned model with PDF context if available"""
    experience_prompt = get_experience_level_prompt(experience_level)
    
    if st.session_state["uploaded_pdf_text"]:
        context = f"Here's the context from the uploaded PDF:\n\n{st.session_state['uploaded_pdf_text']}\n\n"
        full_prompt = f"{context}{query}"
    else:
        full_prompt = query

    try:
        # Using OpenAI client for direct API access
        client = openai.OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        
        response = client.chat.completions.create(
            model="ft:open-mistral-7b:c995defc:20241117:3e25cc8e",
            temperature=0,
            messages=[
                {"role": "system", "content": f"You are a legal AI assistant. {experience_prompt}"},
                {"role": "user", "content": full_prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Fine-tuned model error: {str(e)}. Falling back to base model.")
        # Fallback to base Mistral model using direct API call
        try:
            response = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                temperature=0,
                messages=[
                    {"role": "system", "content": f"You are a legal AI assistant. {experience_prompt}"},
                    {"role": "user", "content": full_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: Unable to generate response. Please try another model or contact support. ({str(e)})"

def call_llama_ft(query, experience_level="Intermediate"):
    """Direct call to Llama fine-tuned model with PDF context if available"""
    experience_prompt = get_experience_level_prompt(experience_level)
    
    if st.session_state["uploaded_pdf_text"]:
        context = f"Here's the context from the uploaded PDF:\n\n{st.session_state['uploaded_pdf_text']}\n\n"
        full_prompt = f"{context}{query}"
    else:
        full_prompt = query

    try:
        client = openai.OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        
        # Enhanced prompt for fine-tuned behavior
        system_prompt = f"""You are an advanced legal AI assistant trained specifically in legal analysis and documentation.
        {experience_prompt}
        
        Guidelines for response:
        1. Focus exclusively on legal aspects
        2. Use appropriate legal terminology for the specified experience level
        3. Cite relevant laws and precedents when applicable
        4. Maintain professional and formal tone
        5. Provide structured, clear explanations
        """
        
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Fine-tuned model error: {str(e)}. Falling back to base model.")
        try:
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                temperature=0,
                messages=[
                    {"role": "system", "content": f"You are a legal AI assistant. {experience_prompt}"},
                    {"role": "user", "content": full_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: Unable to generate response. Please try another model or contact support. ({str(e)})"

def process_pdf_content(pdf_file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        extracted_text = ""
        for page in range(len(pdf_reader.pages)):
            extracted_text += pdf_reader.pages[page].extract_text() or ""
        return extracted_text
    except Exception as e:
        st.sidebar.error(f"Error reading PDF file: {e}")
        return None

def process_and_store_pdf_content(pdf_text):
    """Process PDF content and store in persistent collection"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(pdf_text)

    try:
        # Store in persistent collection
        vectorstore = setup_vectorstore(PERSISTENT_COLLECTION)
        try:
            vectorstore.clear()  # Clear existing embeddings
        except:
            pass
        vectorstore.add_texts(chunks)
        
        st.session_state.vectorstore = vectorstore
        st.session_state.is_doc_mode = True
        return vectorstore
    except Exception as e:
        st.error(f"Error storing PDF content: {str(e)}")
        return None

def handle_user_query(user_input, selected_model, selected_model_name, experience_level):
    """Handle user query with strict experience level differentiation"""
    try:
        experience_prompt = get_experience_level_prompt(experience_level)
        
        # Case 1: Fine-tuned models
        if "ft" in selected_model:
            try:
                if "gpt" in selected_model:
                    response = call_gpt_4o(user_input, experience_level=experience_level)
                elif "gemini" in selected_model:
                    response = call_gemini_ft(user_input, experience_level=experience_level)
                elif "mistral" in selected_model:
                    response = call_mistral_ft(user_input, experience_level=experience_level)
                elif "llama" in selected_model:
                    response = call_llama_ft(user_input, experience_level=experience_level)
                return response, None
            except Exception as e:
                st.warning(f"Fine-tuned model error: {str(e)}. Falling back to base model.")
                base_model = selected_model.split('-')[0] + "-base"
                return handle_user_query(user_input, base_model, f"{selected_model_name} (Fallback)", experience_level)

        # Case 2: RAG Models
        elif "rag" in selected_model:
            # For non-document mode, use llamarag_collections
            if not st.session_state.is_doc_mode:
                vectorstore = setup_vectorstore("llamarag_collections")
                if vectorstore is None:
                    return "I don't have enough information to answer this question.", None
                    
                # Get top 5 relevant documents
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                docs = retriever.get_relevant_documents(user_input)
                
                if not docs:
                    return "I don't have enough information to answer this question.", None
                
            # For document mode, use persistent collection
            else:
                if st.session_state.vectorstore is None:
                    return "No documents have been uploaded yet.", None
                    
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                docs = retriever.get_relevant_documents(user_input)
            
            # Format context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            enhanced_prompt = f"""
            Experience Level: {experience_level}
            
            Use the following context to answer the question. If the answer cannot be found in the context, 
            respond with "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Follow these rules while answering:
            {experience_prompt}
            
            Question: {user_input}
            """

            # Handle different RAG models
            try:
                if "llama-rag" in selected_model or "mistral-rag" in selected_model:
                    # Determine correct Groq model
                    model_name = "mixtral-8x7b-32768" if "mistral" in selected_model else "llama-3.1-70b-versatile"
                    client = openai.OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
                    response = client.chat.completions.create(
                        model=model_name,
                        temperature=0,
                        messages=[
                            {"role": "system", "content": "You are a legal AI assistant."},
                            {"role": "user", "content": enhanced_prompt}
                        ]
                    )
                    answer = response.choices[0].message.content.strip()
                
                elif "gemini-rag" in selected_model:
                    genai.configure(api_key=GOOGLE_API_KEY)
                    model = genai.GenerativeModel('gemini-1.5-pro')
                    response = model.generate_content([
                        {"role": "user", "parts": ["You are a legal AI assistant."]},
                        {"role": "model", "parts": ["I understand. I will act as a legal AI assistant."]},
                        {"role": "user", "parts": [enhanced_prompt]}
                    ])
                    answer = response.text.strip()
                
                elif "gpt-rag" in selected_model:
                    client = openai.OpenAI(api_key=OPENAI_API_KEY)
                    response = client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        temperature=0,
                        messages=[
                            {"role": "system", "content": "You are a legal AI assistant."},
                            {"role": "user", "content": enhanced_prompt}
                        ]
                    )
                    answer = response.choices[0].message.content.strip()
                
                else:
                    raise ValueError(f"Unsupported RAG model: {selected_model}")

                sources = "\n\n".join([f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
                return answer, sources
                
            except Exception as e:
                st.error(f"Error with RAG model: {str(e)}")
                return "I apologize, but I encountered an error while processing your request with the RAG model.", None
        
        # Case 3: Base model
        else:
            if st.session_state.conversation_chain is None:
                st.session_state.conversation_chain = create_base_chain(
                    selected_model,
                    experience_level
                )
            
            enhanced_prompt = f"""
            {experience_prompt}
            
            Strictly follow the above rules while answering this question:
            {user_input}
            
            Remember to maintain the appropriate {experience_level} level in your response.
            """
            
            response = st.session_state.conversation_chain.predict(input=enhanced_prompt)
            return response.strip(), None
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request.", None

def check_password(username, password):
    """Verify user credentials"""
    if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
        return True
    stored_password_hash = st.secrets.get("users", {}).get(username)
    if stored_password_hash:
        return bcrypt.checkpw(password.encode(), stored_password_hash.encode())
    return False

def clear_document_mode():
    """Clear all document-related state and reset to base mode"""
    if "conversation_chain" in st.session_state:
        del st.session_state.conversation_chain
    if "vectorstore" in st.session_state:
        del st.session_state.vectorstore
    if "uploaded_pdf_text" in st.session_state:
        del st.session_state.uploaded_pdf_text
    st.session_state.is_doc_mode = False

def logout():
    """Handle user logout"""
    keys_to_clear = [
        "logged_in", "chat_history", "conversation_chain", "vectorstore",
        "uploaded_pdf_text", "current_model", "is_doc_mode", "model_type",
        "token", "temporary_collection_name", "experience_level",
        "last_experience_level"
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def login():
    """Handle user login"""
    col1, col2 = st.columns([1, 10], gap='small')
    with col1:
        st.image("Logo2.png", width=60)
    with col2:
        st.title("LexLLM")

    if "token" in st.session_state:
        try:
            jwt_response = descope_client.validate_and_refresh_session(
                st.session_state.token, 
                st.session_state.refresh_token
            )
            st.session_state["token"] = jwt_response["sessionToken"].get("jwt")
            st.session_state["logged_in"] = True
            return
        except Exception:
            pass

    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_password(username, password):
            st.session_state["logged_in"] = True
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    if "code" in st.query_params:
        code = st.query_params["code"]
        st.query_params.clear()
        try:
            with st.spinner("Loading..."):
                jwt_response = descope_client.sso.exchange_token(code)
                st.session_state["token"] = jwt_response["sessionToken"].get("jwt")
                st.session_state["refresh_token"] = jwt_response["refreshSessionToken"].get("jwt")
                st.session_state["user"] = jwt_response["user"]
                st.session_state["logged_in"] = True
                st.rerun()
        except AuthException as e:
            st.error(f"Login failed: {str(e)}")

    with st.container(border=True):
        if st.button("🅖 Continue with Google", use_container_width=True):
            oauth_response = descope_client.oauth.start(
                provider="google",
                return_url="http://localhost:8501"
            )
            url = oauth_response["url"]
            st.markdown(
                f'<meta http-equiv="refresh" content="0; url={url}">',
                unsafe_allow_html=True,
            )

def main():
    """Main application function"""
    if not st.session_state.get("logged_in", False):
        login()
        return
        
    try:
        if "token" in st.session_state:
            jwt_response = descope_client.validate_and_refresh_session(
                st.session_state.token, 
                st.session_state.refresh_token
            )
            st.session_state["token"] = jwt_response["sessionToken"].get("jwt")
    except AuthException:
        st.warning("Session expired. Please login again.")
        st.session_state["logged_in"] = False
        st.rerun()
        return

    # Main UI
    col1, col2 = st.columns([1, 10], gap='small')
    with col1:
        st.image("Logo2.png", width=60)
    with col2:
        st.header("LexLLM")

    # Experience level selection
    experience_level = st.sidebar.selectbox(
        "Choose your legal document experience",
        ("Beginner", "Intermediate", "Advanced"),
        key='experience_level'
    )

    # Model selection with categories
    model_options = {
        "Base Models": {
            "Llama Base": "llama-base",
            "GPT Base": "gpt-base",
            "Gemini Base": "gemini-base",
            "Mistral Base": "mistral-base"
        },
        "RAG Models": {
            "Llama RAG": "llama-rag",
            "GPT RAG": "gpt-rag",
            "Gemini RAG": "gemini-rag",
            "Mistral RAG": "mistral-rag"
        },
        "Fine-Tuned Models": {
            "GPT-4o Fine-Tuned": "gpt-4o-ft",
            "Gemini Fine-Tuned": "gemini-ft",
            "Mistral Fine-Tuned": "mistral-ft",
            "Llama Fine-Tuned": "llama-ft"
        }
    }

    model_category = st.sidebar.radio(
        "Select Model Category",
        options=list(model_options.keys())
    )

    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options[model_category].keys())
    )
    selected_model = model_options[model_category][selected_model_name]

    # Warning for fine-tuned models
    if "ft" in selected_model:
        st.sidebar.warning("Note: Fine-tuned model access requires special permissions. The system will fallback to base model if unavailable.")

    # Show active model
    st.sidebar.info(f"Active Model: {selected_model_name}")

    # Mode indicator
    mode_text = "Document Mode" if st.session_state.is_doc_mode else "Base Knowledge Mode"
    st.sidebar.info(f"Current Mode: {mode_text}")

    # Debug information
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.markdown("### Debug Information")
        st.sidebar.markdown("---")
        
        st.sidebar.write("**Model Information:**")
        st.sidebar.write(f"- Current Model: {selected_model}")
        st.sidebar.write(f"- Model Category: {model_category}")
        
        st.sidebar.write("\n**Chain Information:**")
        chain_type = type(st.session_state.conversation_chain).__name__ if st.session_state.conversation_chain else "None"
        st.sidebar.write(f"- Chain Type: {chain_type}")
        
        st.sidebar.write("\n**Document Mode:**")
        st.sidebar.write(f"- PDF Mode Active: {st.session_state.is_doc_mode}")
        
        if st.session_state.is_doc_mode:
            st.sidebar.write("\n**Collections:**")
            st.sidebar.write(f"- Collection: {'llamarag_collections' if 'llama-rag' in selected_model else 'persistent_docs_collection'}")
            
            if st.session_state.vectorstore:
                st.sidebar.write("\n**Vectorstore Status:**")
                st.sidebar.write("- Vectorstore: Initialized")
                try:
                    docs = st.session_state.vectorstore.similarity_search("test", k=1)
                    st.sidebar.write("- Documents: Available")
                except:
                    st.sidebar.write("- Documents: None")
            else:
                st.sidebar.write("\n**Vectorstore Status:**")
                st.sidebar.write("- Vectorstore: Not initialized")

    # PDF upload functionality
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            pdf_text = process_pdf_content(uploaded_file)
            if pdf_text:
                st.session_state.uploaded_pdf_text = pdf_text
                vectorstore = process_and_store_pdf_content(pdf_text)
                if vectorstore:
                    st.session_state.conversation_chain = create_chain_with_retrieval(
                        selected_model, 
                        vectorstore,
                        experience_level
                    )
                    st.sidebar.success("PDF processed and stored successfully!")

    # Option to clear document mode
    if st.session_state.is_doc_mode:
        if st.sidebar.button("Clear Document Mode"):
            clear_document_mode()
            st.rerun()

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])
            if "model" in message:
                st.caption(f"Response generated by: {message['model']} (Experience Level: {message['experience_level']})")
            if "sources" in message and message["sources"]:
                with st.expander("View source documents"):
                    st.markdown(message["sources"])

    # Chat input
    user_input = st.chat_input("Ask your legal question...")
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        st.chat_message("user").markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner(f"Generating response using {selected_model_name}..."):
                try:
                    assistant_response, sources = handle_user_query(
                        user_input,
                        selected_model,
                        selected_model_name,
                        experience_level
                    )
                    
                    # Display response
                    st.markdown(assistant_response)
                    st.caption(f"Response generated by: {selected_model_name} (Experience Level: {experience_level})")
                    
                    # Display sources if available
                    if sources:
                        with st.expander("View source documents"):
                            st.markdown(sources)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_response,
                        "model": selected_model_name,
                        "experience_level": experience_level,
                        "sources": sources if sources else None
                    })

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.error("Please try selecting a different model or refreshing the page.")

    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("Log out"):
        logout()

if __name__ == "__main__":
    init_session_state()
    main()