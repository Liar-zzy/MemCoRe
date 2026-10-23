from typing import List, Dict, Optional, Literal, Any, Union, Tuple
import json
from datetime import datetime
import uuid
import numpy as np
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from nltk.tokenize import word_tokenize
import pickle
import time
from memcore.versions.v1.prompts.prompt_loader import PromptLoader
from memcore.versions.v1.prompts.analyze_content_prompt import GRAPH_STOP_WORDS


def completion(**kwargs):
    """Load LiteLLM only when the optional Ollama controller is used."""
    from litellm import completion as litellm_completion
    return litellm_completion(**kwargs)

def simple_tokenize(text):
    return word_tokenize(text)

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass


def get_json_completion_with_retry(
    controller,
    prompt,
    response_format,
    temperature=0.0,
    max_tokens=8192,
    attempts=3,
    retry_delay=1.0,
):
    """Request and parse structured JSON, retrying truncation/API failures."""
    last_error = None
    for attempt in range(attempts):
        try:
            response = controller.get_completion(
                prompt,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return json.loads(response)
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                print(
                    f"      Structured completion retry {attempt + 1}/{attempts}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                if retry_delay:
                    time.sleep(retry_delay * (2 ** attempt))
    raise RuntimeError(
        f"structured completion failed after {attempts} attempts: {last_error}"
    ) from last_error

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, base_url: Optional[str] = None, extra_body: Optional[dict] = None):
        from openai import OpenAI
        self.model = model
        self.extra_body = extra_body
        # Local OpenAI-compatible servers may accept a placeholder key.
        self.client = OpenAI(
            api_key=api_key if api_key is not None else os.getenv('OPENAI_API_KEY', 'empty'),
            base_url=base_url,
        )

        # E1: Token tracking support
        self._token_tracker = None
        self._current_caller = "unknown"
        self._last_completion_metadata = None

        # Not every OpenAI-compatible provider implements structured outputs.
        # vLLM does; DeepSeek rejects json_schema with a 400 and only accepts
        # json_object. Probed on first use, then remembered.
        self._json_schema_supported = True

    @property
    def last_completion_metadata(self):
        """Diagnostics for the last response, without retaining its content."""
        return (dict(self._last_completion_metadata)
                if self._last_completion_metadata is not None else None)

    @staticmethod
    def _downgrade_response_format(response_format: dict) -> Tuple[dict, str]:
        """Turn a json_schema response_format into plain json_object mode.

        The schema is no longer enforced by the provider, so it is handed to the
        model as text instead -- otherwise it would be free to invent key names.
        Returns (new_response_format, extra_system_text).
        """
        if not response_format or response_format.get("type") != "json_schema":
            return response_format, ""
        schema = response_format.get("json_schema", {}).get("schema")
        if schema is None:
            return {"type": "json_object"}, ""
        return {"type": "json_object"}, (
            " Your reply must validate against this JSON schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )

    def _prepare_response_format(self, response_format):
        """GPT-4o strict outputs require every declared object field, including link reasons."""
        if not (self.model.startswith("gpt-4o") and response_format
                and response_format.get("type") == "json_schema"
                and response_format.get("json_schema", {}).get("strict")):
            return response_format
        prepared = deepcopy(response_format)

        def require_fields(schema):
            if isinstance(schema, dict):
                if schema.get("type") == "object":
                    schema["required"] = list(schema.get("properties", {}))
                for value in schema.values():
                    require_fields(value)
            elif isinstance(schema, list):
                for value in schema:
                    require_fields(value)

        require_fields(prepared["json_schema"]["schema"])
        return prepared

    def get_completion(self, prompt: str, response_format: Optional[dict] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None) -> str:
        self._last_completion_metadata = None
        def build_kwargs(fmt, downgraded):
            messages = [{"role": "user", "content": prompt}]
            if fmt:
                system_text = "You must respond with a JSON object."
                template_options = (self.extra_body or {}).get("chat_template_kwargs", {})
                if template_options.get("enable_thinking") is False:
                    system_text += " /no_think"
                if downgraded:
                    _, schema_text = self._downgrade_response_format(response_format)
                    system_text += schema_text
                messages.insert(0, {"role": "system", "content": system_text})

            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if fmt:
                kwargs["response_format"] = fmt
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body
            return kwargs

        wants_schema = bool(response_format) and response_format.get("type") == "json_schema"
        use_downgraded = wants_schema and not self._json_schema_supported
        active_format = self._prepare_response_format(response_format)
        if use_downgraded:
            active_format, _ = self._downgrade_response_format(response_format)

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                **build_kwargs(active_format, use_downgraded))
        except Exception as exc:
            # Retry once in json_object mode if the provider rejected json_schema.
            if not (wants_schema and not use_downgraded
                    and "response_format" in str(exc)):
                raise
            self._json_schema_supported = False
            active_format, _ = self._downgrade_response_format(response_format)
            response = self.client.chat.completions.create(
                **build_kwargs(active_format, True))
        wall_time = time.time() - start_time

        content = response.choices[0].message.content
        self._last_completion_metadata = {
            'output_characters': len(content) if isinstance(content, str) else 0,
            'finish_reason': getattr(response.choices[0], 'finish_reason', None),
            'usage_recorded': bool(self._token_tracker and response.usage),
        }

        # E1: Record token usage if tracker is set
        if self._token_tracker and response.usage:
            self._token_tracker.record(
                caller=self._current_caller,
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                wall_time_s=wall_time,
                model=self.model,
            )

        return content

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        self.model = model
        self._last_completion_metadata = None

    @property
    def last_completion_metadata(self):
        return (dict(self._last_completion_metadata)
                if self._last_completion_metadata is not None else None)
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: Optional[dict] = None,
                       temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        self._last_completion_metadata = None
        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            if response_format:
                messages.insert(0, {"role": "system", "content": "You must respond with a JSON object."})
            
            kwargs = {
                "model": "ollama_chat/{}".format(self.model),
                "messages": messages,
            }
            if response_format:
                kwargs["response_format"] = response_format
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            
            response = completion(**kwargs)
            content = response.choices[0].message.content
            self._last_completion_metadata = {
                'output_characters': len(content) if isinstance(content, str) else 0,
                'finish_reason': getattr(response.choices[0], 'finish_reason', None),
                'usage_recorded': False,
            }
            return content
        except Exception as e:
            if response_format:
                empty_response = self._generate_empty_response(response_format)
                content = json.dumps(empty_response)
                self._last_completion_metadata = {
                    'output_characters': len(content), 'finish_reason': None,
                    'usage_recorded': False, 'error_type': type(e).__name__,
                }
                return content
            self._last_completion_metadata = {
                'output_characters': 0, 'finish_reason': None,
                'usage_recorded': False, 'error_type': type(e).__name__,
            }
            return ""

class EmbeddingController:
    """独立的 Embedding 控制器"""
    def __init__(self, 
                 backend: str = "openai",
                 model: str = "text-embedding-3-small", 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.backend = backend
        self.model = model
        self.last_embedding_usage = None
        
        if backend == "openai":
            try:
                from openai import OpenAI
                if api_key is None:
                    api_key = 'empty'
                if base_url is None:
                    base_url = ""
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
            except ImportError:
                raise ImportError("OpenAI package not found. Install it with: pip install openai")
        else:
            raise ValueError("Currently only 'openai' backend is supported for embeddings")
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的 embedding 向量"""
        self.last_embedding_usage = None
        if self.backend == "openai":
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            usage = getattr(response, 'usage', None)
            prompt_tokens = getattr(usage, 'prompt_tokens', None) if usage is not None else None
            total_tokens = getattr(usage, 'total_tokens', None) if usage is not None else None
            self.last_embedding_usage = {
                'input_tokens': prompt_tokens if type(prompt_tokens) is int else None,
                'total_tokens': total_tokens if type(total_tokens) is int else None,
                'call_count': 1,
            }
            return response.data[0].embedding
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def get_dimension(self) -> int:
        """返回 embedding 的维度"""
        # 根据模型返回对应维度
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "Qwen/Qwen3-Embedding-8B": 4096,
            "Qwen/Qwen3-Embedding-0.6B": 1024
        }
        return dimension_map.get(self.model, 1536)  # 默认 1536

class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self, 
                 backend: Literal["openai", "ollama"] = "openai",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 extra_body: Optional[dict] = None):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key, base_url, extra_body)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        else:
            raise ValueError("Backend must be either 'openai' or 'ollama'")

class MemoryNote:
    """Basic memory unit with metadata"""
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None, 
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[LLMController] = None,
                 use_v2: bool = False):
        
        self.content = content
        
        # Generate metadata using LLM if not provided and controller is available
        if llm_controller and any(param is None for param in [keywords, context, category, tags]):
            if use_v2:
                analysis = self.analyze_content_v2(content, llm_controller)
            else:
                analysis = self.analyze_content(content, llm_controller)
            print(f"V2: {use_v2} analysis: {analysis}")
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]
        
        # Set default values for optional parameters
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Handle context that can be either string or list
        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)  # Convert list to string by joining
            
        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []

    @staticmethod
    def analyze_content(content: str, llm_controller: LLMController) -> Dict:            
        """Analyze content to extract keywords, context, and other metadata"""
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = llm_controller.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                            },
                            "required": ["keywords", "context", "tags"],
                            "additionalProperties": False
                        },
                        "strict": True
                }
            })
            
            try:
                analysis = json.loads(response)
            except:
                print(f"Warning: Failed to parse LLM response as JSON: {response}")
                analysis = {
                    "keywords": [],
                    "context": "General",
                    "tags": []
                }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing content: {str(e)}")
            return {
                "keywords": [],
                "context": "General",
                "category": "Uncategorized",
                "tags": []
            }
    
    @staticmethod
    def analyze_content_v2(content: str, llm_controller: LLMController, prompt_loader: PromptLoader) -> Dict:
        
        # prompt = analyze_content_prompt_v3(content)
        prompt = prompt_loader.get_analyze_content_prompt()(content)

        try:
            response = llm_controller.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                }
                            },
                            "required": ["keywords", "context"],
                            "additionalProperties": False
                        },
                        "strict": True
                }
            })
            
            try:
                analysis = json.loads(response)
            except:
                # analysis = response
                analysis = {
                    "keywords": [],
                    "context": str(response)[:200]  # 截取一部分作为上下文
                }
            
            # =================== 开始去重与清洗逻辑 ===================
            
            # 获取原始列表，防止 key 不存在的情况
            raw_keywords = analysis.get("keywords", [])
            
            # 辅助函数：清洗单个词（去除首尾空格）
            def clean_term(term):
                return str(term).strip() if term else ""

            # --- Step 1: 处理 Keywords ---
            # 使用 set 来记录已存在的小写形式，用于去重
            seen_keywords_lower = set()
            final_keywords = []
            
            for k in raw_keywords:
                clean_k = clean_term(k)
                if not clean_k: continue
                
                k_lower = clean_k.lower()

                if k_lower in GRAPH_STOP_WORDS:
                    continue

                # 仅当该词的小写形式未出现过时添加 (保留原始大小写)
                if k_lower not in seen_keywords_lower:
                    seen_keywords_lower.add(k_lower)
                    final_keywords.append(clean_k)

            # --- Step 2: 更新 analysis 对象 ---
            analysis["keywords"] = final_keywords
            # 确保 context 存在
            if "context" not in analysis:
                analysis["context"] = "General context."

            # =================== 结束去重与清洗逻辑 ===================

            return analysis
            
        except Exception as e:
            print(f"Error analyzing content: {str(e)}")
            return {
                "keywords": [],
                "context": "General"
            }

class SimpleEmbeddingRetriever:
    """Simple retrieval system using only text embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the simple embedding retriever.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # Map document content to its index
        
    def add_documents(self, documents: List[str]):
        """Add documents to the retriever."""
        # Reset if no existing documents
        if not self.corpus:
            self.corpus = documents
            # print("documents", documents, len(documents))
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            # Append new documents
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, float]]:
        """Search for similar documents using cosine similarity.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of dicts with document text and score
        """
        if not self.corpus:
            return []
        # print("corpus", len(self.corpus), self.corpus)
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # Get top k results
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return top_k_indices
        
    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk"""
        # Save embeddings using numpy
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
        
        # Save other attributes
        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk"""
        print(f"Loading retriever from {retriever_cache_file} and {retriever_cache_embeddings_file}")
        
        # Load embeddings
        if os.path.exists(retriever_cache_embeddings_file):
            print(f"Loading embeddings from {retriever_cache_embeddings_file}")
            self.embeddings = np.load(retriever_cache_embeddings_file)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            print(f"Embeddings file not found: {retriever_cache_embeddings_file}")
        
        # Load other attributes
        if os.path.exists(retriever_cache_file):
            print(f"Loading corpus from {retriever_cache_file}")
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
                print(f"Loaded corpus with {len(self.corpus)} documents")
        else:
            print(f"Corpus file not found: {retriever_cache_file}")
            
        return self

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str) -> 'SimpleEmbeddingRetriever':
        """Load retriever state from memory"""
        # Create documents combining content and metadata for each memory
        all_docs = []
        for m in memories.values():
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)
            
        # Create and initialize retriever
        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever

class AgenticMemorySystem:
    """Memory management system with embedding-based retrieval"""
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 extra_body: Optional[dict] = None):
        self.memories = {}  # id -> MemoryNote
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, base_url, extra_body)
        self.evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                
                                Important notes about tags:
                                - tags_to_update: Contains tags for the NEW memory (current memory being added). Each tag should be unique within this list (no duplicates).
                                - new_tags_neighborhood: Contains tags for EACH neighbor memory. This is a list of lists, where each inner list corresponds to one neighbor memory. Each inner list should have unique tags (no duplicates within that list).
                                - Different memories can have similar or identical tags if their content is similar - this is acceptable and helps with retrieval.
                                - Avoid duplicate tags within the same tag list (e.g., ["tag1", "tag1", "tag2"] should be ["tag1", "tag2"]).
                                
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''
        self.evo_cnt = 0 
        self.evo_threshold = evo_threshold

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note"""
        note = MemoryNote(content=content, llm_controller=self.llm_controller, timestamp=time, **kwargs)
        
        # Update retriever with all documents
        # all_docs = [m.content for m in self.memories.values()]
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_documents(["content:" + note.content + " context:" + note.context + " keywords: " + ", ".join(note.keywords) + " tags: " + ", ".join(note.tags)])
        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id
    
    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents
        
        This function re-initializes the retriever and updates it with all memory documents,
        including their context, keywords, and tags to ensure the retrieval system has the
        latest state of all memories.
        """
        # Reset the retriever with the same model
        try:
            # Try to get model name through get_config_dict if available
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            # Fallback: use the model name from the class initialization
            model_name = 'all-MiniLM-L6-v2'
        
        self.retriever = SimpleEmbeddingRetriever(model_name)
        
        # Re-add all memory documents with their metadata
        for memory in self.memories.values():
            # Combine memory metadata into a single searchable document
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            # Add both the content and metadata as separate documents for better retrieval
            self.retriever.add_documents([memory.content + " , " + metadata_text])
    
    def process_memory(self, note: MemoryNote) -> bool:
        """Process a memory note and return an evolution label"""
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        
        # 将indices转换为memory_ids列表，供后续使用
        # all_memories_list = list(self.memories.values())
        # all_memory_ids = list(self.memories.keys())
        
        # 如果没有邻居记忆，直接返回
        if not indices:
            return False, note
        
        prompt_memory = self.evolution_system_prompt.format(context=note.context, content=note.content, keywords=note.keywords, nearest_neighbors_memories=neighbor_memory,neighbor_number=len(indices))
        # print("prompt_memory", prompt_memory)
        response = self.llm_controller.llm.get_completion(
            prompt_memory,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean",
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "integer"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve","actions","suggested_connections","tags_to_update","new_context_neighborhood","new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
        )
        # try:
        print("response: ", response)
        response_json = json.loads(response)
        # print("response_json", response_json, type(response_json))
        # except:
        #     response_json = response
        should_evolve = response_json["should_evolve"]
        if should_evolve:
            actions = response_json["actions"]
            for action in actions:
                if action == "strengthen":
                    suggest_connections = response_json["suggested_connections"]
                    new_tags = response_json["tags_to_update"]
                    print('links', note.links)
                    note.links.extend(suggest_connections)
                    print('Update ->', note.links)
                    note.tags = new_tags

                elif action == "update_neighbor":
                    new_context_neighborhood = response_json["new_context_neighborhood"]
                    new_tags_neighborhood = response_json["new_tags_neighborhood"]
                    noteslist = list(self.memories.values())
                    notes_id = list(self.memories.keys())
                    print("indices", indices)
                    # if slms output less than the number of neighbors, use the sequential order of new tags and context.
                    for i in range(min(len(indices), len(new_tags_neighborhood))):
                        # find some memory
                        tag = new_tags_neighborhood[i]
                        if i < len(new_context_neighborhood):
                            context = new_context_neighborhood[i]
                        else:
                            context = noteslist[indices[i]].context
                        memorytmp_idx = indices[i]
                        notetmp = noteslist[memorytmp_idx]
                        # add tag to memory
                        notetmp.tags = tag
                        notetmp.context = context
                        self.memories[notes_id[memorytmp_idx]] = notetmp
        return should_evolve, note

    def find_related_memories(self, query: str, k: int = 5) -> List[MemoryNote]:
        """Find related memories using hybrid retrieval"""
        if not self.memories:
            return "",[]
            
        # Get indices of related memories
        # indices = self.retriever.retrieve(query_note.content, k)
        indices = self.retriever.search(query, k)
        
        # Convert to list of memories
        all_memories = list(self.memories.values())
        memory_str = ""
        # print("indices", indices)
        # print("all_memories", all_memories)
        
        # Filter valid indices to avoid IndexError
        valid_indices = [i for i in indices if i < len(all_memories)]
        
        for i in valid_indices:
            memory_str += "memory index:" + str(i) + "\t talk start time:" + all_memories[i].timestamp + "\t memory content: " + all_memories[i].content + "\t memory context: " + all_memories[i].context + "\t memory keywords: " + str(all_memories[i].keywords) + "\t memory tags: " + str(all_memories[i].tags) + "\n"
        return memory_str, valid_indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> List[MemoryNote]:
        """Find related memories using hybrid retrieval"""
        if not self.memories:
            return []
            
        # Get indices of related memories
        # indices = self.retriever.retrieve(query_note.content, k)
        indices = self.retriever.search(query, k)
        
        # Convert to list of memories
        all_memories = list(self.memories.values())
        memory_str = ""
        j = 0
        
        # Filter valid indices to avoid IndexError
        valid_indices = [i for i in indices if i < len(all_memories)]
        
        for i in valid_indices:
            memory_str +=  "talk start time:" + all_memories[i].timestamp + "memory content: " + all_memories[i].content + "memory context: " + all_memories[i].context + "memory keywords: " + str(all_memories[i].keywords) + "memory tags: " + str(all_memories[i].tags) + "\n"
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                # Also check neighbor index
                if neighbor < len(all_memories):
                    memory_str += "talk start time:" + all_memories[neighbor].timestamp + "memory content: " + all_memories[neighbor].content + "memory context: " + all_memories[neighbor].context + "memory keywords: " + str(all_memories[neighbor].keywords) + "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                if j >=k:
                    break
                j += 1
        return memory_str

