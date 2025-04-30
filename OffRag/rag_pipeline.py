import time
import logging
import chromadb
import embedchain
import streamlit as st
from embedchain.app import App
from datetime import datetime
import numpy as np
import hashlib
from config import APP_VERSION
import re

logger = logging.getLogger("rag_assistant")

class DebugApp(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_info = {
            "add_operations": [],
            "query_operations": [],
            "current_session": {
                "chunks": [],
                "embeddings": [],
                "retrieved_docs": [],
                "contexts": [],
                "prompt": "",
                "response": "",
                "streaming_info": {"chunk_count": 0, "response_type": "", "chunks_received": []},
            },
        }
        self._embedding_dimension = None

    @property
    def embedder(self):
        if hasattr(self, '_embedder'):
            return self._embedder
        try:
            if hasattr(self, 'db') and hasattr(self.db, 'embedder'):
                self._embedder = self.db.embedder
                if st.session_state.debug_enabled:
                    logger.debug("Using embedder from self.db.embedder")
            elif hasattr(self, 'config') and 'embedder' in self.config:
                from embedchain.embedder.ollama import OllamaEmbedder
                embedder_config = self.config['embedder'].get('config', {})
                self._embedder = OllamaEmbedder(**embedder_config)
                if st.session_state.debug_enabled:
                    logger.debug("Created embedder from config")
            else:
                from embedchain.embedder.ollama import OllamaEmbedder
                self._embedder = OllamaEmbedder(
                    model="nomic-embed-text:latest",
                    base_url="http://localhost:11434"
                )
                if st.session_state.debug_enabled:
                    logger.debug("Created default embedder")
            return self._embedder
        except Exception as e:
            logger.error(f"Failed to get/create embedder: {str(e)}")
            raise

    @embedder.setter
    def embedder(self, value):
        self._embedder = value

    def _compute_embedding_dimension(self, text="test"):
        if self._embedding_dimension is None:
            try:
                embeddings = self.embedder.to_embeddings([text])
                embedding = embeddings[0] if isinstance(embeddings, list) else embeddings
                self._embedding_dimension = len(embedding)
            except Exception as e:
                logger.error(f"Failed to compute embedding dimension: {str(e)}")
                self._embedding_dimension = 0
        return self._embedding_dimension

    def _generate_embedding_id(self, text, file_name="Unknown"):
        content = f"{file_name}:{text}"
        return f"default-app-id--{hashlib.sha256(content.encode('utf-8')).hexdigest()}"

    def add(self, *args, debug_mode=False, file_name="Unknown", **kwargs):
        start_time = time.time()
        data_type = kwargs.get("data_type", "unknown")
        text_snippet = str(args[0])[:100] + "..." if args and args[0] else "No text"
        if debug_mode:
            logger.info(f"Adding document (type={data_type}, file_name={file_name}): snippet='{text_snippet}'")

        try:
            if not args or not args[0].strip():
                raise ValueError("Empty text provided to embed")
            text = args[0]
            embedding_id = self._generate_embedding_id(text, file_name)
            try:
                collection = self.db.client.get_collection("default")
                existing = collection.get(ids=[embedding_id])
                if existing['ids']:
                    if debug_mode:
                        logger.info(f"Skipping duplicate embedding ID: {embedding_id}")
                    duration = time.time() - start_time
                    operation_info = {
                        "timestamp": datetime.now().isoformat(),
                        "data_type": data_type,
                        "file_name": file_name,
                        "text_snippet": text_snippet,
                        "duration": duration,
                        "success": True,
                        "embedding_dimension": self._compute_embedding_dimension(),
                        "status": "Skipped (Duplicate)",
                        "total_chunks": 1,
                        "successful_chunks": 0,
                        "total_duration": duration,
                        "avg_duration": duration,
                    }
                    self.debug_info["add_operations"].append(operation_info)
                    return operation_info
            except Exception as e:
                if debug_mode:
                    logger.warning(f"Error checking existing embedding ID {embedding_id}: {str(e)}")

            embedding_dimension = self._compute_embedding_dimension() if debug_mode else 0
            result = super().add(text, **kwargs)
            if result is None:
                raise ValueError("Embedding result is None")
            duration = time.time() - start_time
            operation_info = {
                "timestamp": datetime.now().isoformat(),
                "data_type": data_type,
                "file_name": file_name,
                "text_snippet": text_snippet,
                "duration": duration,
                "success": True,
                "embedding_dimension": embedding_dimension,
                "status": "Success",
                "total_chunks": 1,
                "successful_chunks": 1,
                "total_duration": duration,
                "avg_duration": duration,
            }
        except Exception as e:
            logger.error(f"Error in add method: {e}")
            duration = time.time() - start_time
            operation_info = {
                "timestamp": datetime.now().isoformat(),
                "data_type": data_type,
                "file_name": file_name,
                "text_snippet": text_snippet,
                "duration": duration,
                "success": False,
                "error": str(e),
                "embedding_dimension": 0,
                "status": "Failed",
                "total_chunks": 1,
                "successful_chunks": 0,
                "total_duration": duration,
                "avg_duration": duration,
            }
        if debug_mode:
            logger.debug(f"Operation info created: {operation_info}")
        self.debug_info["add_operations"].append(operation_info)
        if len(self.debug_info["add_operations"]) > 100:
            self.debug_info["add_operations"] = self.debug_info["add_operations"][-100:]
        return operation_info

    def chat(self, prompt, debug_mode=False, **kwargs):
        try:
            if debug_mode:
                logger.debug(f"Chat called with prompt: '{prompt[:50]}...'")
            self.debug_info["current_session"] = {
                "prompt": prompt,
                "response": "",
                "retrieved_docs": [],
                "contexts": [],
                "chunks": [],
                "embeddings": [],
                "streaming_info": {"chunk_count": 0, "response_type": "", "chunks_received": []},
            }
            start_time = time.time()
            chunk_count = 0
            kwargs = {k: v for k, v in kwargs.items() if k != 'stream'}
            response = super().chat(prompt, stream=True, **kwargs)
            if debug_mode:
                response_type = type(response).__name__
                self.debug_info["current_session"]["streaming_info"]["response_type"] = response_type
                logger.debug(f"super().chat response type: {response_type}")
            if not hasattr(response, '__iter__') or isinstance(response, (str, bytes)):
                if debug_mode:
                    logger.warning(f"Non-iterable response from super().chat: {type(response).__name__}, wrapping as generator")
                response = iter([str(response)])
            for chunk in response:
                chunk_count += 1
                if isinstance(chunk, dict):
                    chunk = chunk.get('text', '')
                elif not isinstance(chunk, str):
                    chunk = str(chunk)
                if debug_mode:
                    self.debug_info["current_session"]["streaming_info"]["chunks_received"].append({
                        "chunk_number": chunk_count,
                        "content": chunk[:50] + "..." if len(chunk) > 50 else chunk,
                        "length": len(chunk),
                        "timestamp": time.time() - start_time,
                    })
                    logger.debug(
                        f"Raw chunk {chunk_count} from super().chat at {time.time() - start_time:.2f}s: "
                        f"'{chunk[:50]}...' (len={len(chunk)})"
                    )
                filtered_chunk = re.sub(r"<think>.*?</think>", "", chunk, flags=re.DOTALL)
                if filtered_chunk.strip():
                    if debug_mode:
                        logger.debug(
                            f"Yielding filtered chunk {chunk_count} at {time.time() - start_time:.2f}s: "
                            f"'{filtered_chunk[:50]}...' (len={len(filtered_chunk)})"
                        )
                    yield filtered_chunk
                    self.debug_info["current_session"]["response"] += filtered_chunk
            if debug_mode:
                self.debug_info["current_session"]["streaming_info"]["chunk_count"] = chunk_count
                logger.debug(f"Chat response completed in {time.time() - start_time:.2f}s, total chunks: {chunk_count}")
            st.session_state.debug_sessions.append(self.debug_info["current_session"])
            if len(st.session_state.debug_sessions) > 100:
                st.session_state.debug_sessions = st.session_state.debug_sessions[-100:]
            return self.debug_info["current_session"]["response"]
        except Exception as e:
            logger.error(f"Chat error: {e}")
            self.debug_info["current_session"]["response"] = f"Error: {str(e)}"
            st.session_state.debug_sessions.append(self.debug_info["current_session"])
            raise

    def force_reset_db(self, debug_mode=False):
        try:
            if hasattr(self, 'db') and hasattr(self.db, 'client'):
                if debug_mode:
                    logger.debug("Force resetting ChromaDB database")
                client = self.db.client
                collections = client.list_collections()
                for collection in collections:
                    if debug_mode:
                        logger.debug(f"Deleting collection: {collection.name}")
                    client.delete_collection(collection.name)
                try:
                    if hasattr(client, '_system') and client._system is not None:
                        if debug_mode:
                            logger.debug("Stopping ChromaDB client system")
                        client._system.stop()
                        time.sleep(1)
                    if hasattr(client, '_system') and hasattr(client._system, 'persistence'):
                        persistence = client._system.persistence
                        if hasattr(persistence, 'db'):
                            if debug_mode:
                                logger.debug("Closing SQLite connections")
                            persistence.db.close()
                            time.sleep(1)
                except Exception as e:
                    logger.warning(f"Failed to stop client system or close connections: {str(e)}")
                self.db = None
            else:
                logger.warning("No valid DB client found for force reset")
        except Exception as e:
            logger.error(f"Error force resetting ChromaDB: {str(e)}")
            raise

    def test_embedding(self, text):
        try:
            start_time = time.time()
            embeddings = self.embedder.to_embeddings([text])
            embedding = embeddings[0] if isinstance(embeddings, list) else embeddings
            if isinstance(embedding, np.float32):
                if st.session_state.debug_enabled:
                    logger.error("Detected single float embedding, raising TypeError")
                raise TypeError("Embedding is a single float value")
            self.add(text, data_type="text", debug_mode=st.session_state.debug_enabled, file_name="Test Input")
            duration = time.time() - start_time
            if st.session_state.debug_enabled:
                logger.debug(f"Embedding test successful, dimension: {len(embedding)}, duration: {duration:.2f}s")
            return {
                "success": True,
                "embedding_dimension": len(embedding),
                "embedding_sample": embedding[:5].tolist(),
                "duration": duration,
            }
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_debug_info(self):
        return self.debug_info

def embedchain_bot(db_path, debug_mode=False):
    try:
        if debug_mode:
            logger.debug(f"Creating DebugApp with db_path: {db_path}")
        app = DebugApp.from_config(
            config={
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": "granite3.3:2b",
                        "max_tokens": 500,
                        "temperature": 0.5,
                        "stream": True,
                        "base_url": "http://localhost:11434",
                    },
                },
                "vectordb": {
                    "provider": "chroma",
                    "config": {"dir": db_path},
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": "nomic-embed-text:latest",
                        "base_url": "http://localhost:11434",
                    },
                },
            }
        )
        if debug_mode:
            logger.debug("DebugApp created successfully")
        return app
    except Exception as e:
        logger.error(f"Failed to create DebugApp: {str(e)}")
        raise

def get_app(db_path, debug_mode=False):
    try:
        if debug_mode:
            logger.debug(f"Creating app with db_path: {db_path}")
        app = embedchain_bot(db_path=db_path, debug_mode=debug_mode)
        return app
    except Exception as e:
        logger.error(f"Failed to create app in get_app: {str(e)}")
        raise