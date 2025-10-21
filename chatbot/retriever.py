import os
import pickle
import logging
# heavy/optional imports (faiss, numpy, langchain) are imported lazily inside
# the methods that need them so importing this module doesn't fail in
# minimal environments.

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _NumpyFlatL2Index:
    """Lightweight L2 index used when FAISS is unavailable."""

    def __init__(self, dim):
        self._dim = dim
        self._vectors = None

    @property
    def d(self):
        return self._dim

    def add(self, vectors):
        self._ensure_numpy()
        import numpy as np

        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError("Vector batch has incompatible shape for index")

        if self._vectors is None:
            self._vectors = np.ascontiguousarray(vectors, dtype="float32")
        else:
            self._vectors = np.vstack([self._vectors, np.ascontiguousarray(vectors, dtype="float32")])

    def reset(self):
        self._vectors = None

    def search(self, query, top_k):
        self._ensure_numpy()
        import numpy as np

        if query.ndim != 2 or query.shape[1] != self._dim:
            raise ValueError("Query batch has incompatible shape for index")

        if self._vectors is None or self._vectors.size == 0:
            return (
                np.empty((1, 0), dtype="float32"),
                np.empty((1, 0), dtype="int64"),
            )

        distances = np.linalg.norm(self._vectors - query, axis=1)
        order = np.argsort(distances)
        top_idx = order[:top_k]
        dists = distances[top_idx].astype("float32").reshape(1, -1)
        inds = top_idx.astype("int64").reshape(1, -1)

        if dists.shape[1] < top_k:
            pad = top_k - dists.shape[1]
            dists = np.pad(dists, ((0, 0), (0, pad)), constant_values=np.inf)
            inds = np.pad(inds, ((0, 0), (0, pad)), constant_values=-1)

        return dists, inds

    @staticmethod
    def _ensure_numpy():
        try:
            import numpy as np  # noqa: F401
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(
                "numpy is required for the fallback index; install with: pip install numpy"
            ) from exc


class RAGRetriever:
    def __init__(self, vector_store_path="data/vector_store.pkl", embed_batch_size=16):
        self.vector_store_path = vector_store_path
        self.index = None
        self.docs = []
        self.embed_batch_size = embed_batch_size
        self._doc_vectors = None
        self._index_backend = None

        # embedding client placeholders
        self._embed_client = None
        self._embed_type = None
        self._embed_model = None
        self._embed_backend_announced = False
        self._init_embed_client()
        self._log_embed_backend()

        self.load_index()

    def _log_embed_backend(self, *, level=logging.INFO, force=False):
        """Log the currently active embedding backend."""
        if self._embed_backend_announced and not force:
            return

        if self._embed_type and self._embed_model:
            logger.log(level, "Embedding backend active: %s (%s)", self._embed_type, self._embed_model)
        elif self._embed_type:
            logger.log(level, "Embedding backend active: %s", self._embed_type)
        else:
            logger.log(level, "Embedding backend not initialized")

        self._embed_backend_announced = True

    def _init_embed_client(self):
        """
        Detect and initialize an embedding client once.
        Prefers local sentence-transformers first, then tries google.genai as fallback.
        Supports:
          - sentence-transformers local model -> SentenceTransformer.encode(...)
          - google.genai (new client) -> client.models.embed_content(...)
          - google.generativeai (older wrapper) -> gga.embeddings.create(...)
        Sets self._embed_type accordingly and stores client object.
        """
        # Prefer local sentence-transformers first
        if self._init_sentence_transformer(log_if_missing=True):
            return

        # Try new google.genai as fallback
        try:
            from google.genai import client as genai_client
            from google.genai import types
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not set; embedding requests will fail until provided")
                # Continue creating the client so we can raise a clearer error later
                client = genai_client.Client()
            else:
                client = genai_client.Client(api_key=api_key)
            # prefer Gemini embedding model if available
            self._embed_client = client
            self._embed_type = "genai"
            self._embed_model = "gemini-embedding-001"
            self._log_embed_backend(force=True)
            return
        except Exception:
            pass

        # No embedding client available
        self._embed_client = None
        self._embed_type = None
        logger.warning(
            "No embedding client initialized. Install `sentence-transformers` for local embeddings or"
            " `google-genai` (pip install google-genai) for cloud embeddings."
        )

    def _init_sentence_transformer(self, *, log_if_missing=False):
        """Attempt to initialize a local sentence-transformer model."""
        model_name = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            if log_if_missing:
                logger.warning(
                    "sentence-transformers not available. Install with: pip install sentence-transformers"
                )
            return False

        try:
            model = SentenceTransformer(model_name)
        except Exception as exc:
            logger.error("Failed to load sentence-transformer model %s: %s", model_name, exc)
            return False

        self._embed_client = model
        self._embed_type = "sentence-transformer"
        self._embed_model = model_name
        self._log_embed_backend(force=True)
        return True

    def load_index(self):
        if not os.path.exists(self.vector_store_path):
            self.index = None
            self.docs = []
            self._doc_vectors = None
            return

        with open(self.vector_store_path, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict):
            self.docs = payload.get("docs", [])
            self._doc_vectors = payload.get("vectors")
            self._index_backend = payload.get("backend")
        elif isinstance(payload, tuple):
            # Backwards compatibility with older pickle format (index, docs)
            if len(payload) >= 2:
                self.index, self.docs = payload[:2]
                self._doc_vectors = payload[2] if len(payload) > 2 else None
                self._index_backend = "faiss"
                logger.info(
                    "Detected legacy vector store format; consider re-adding documents to refresh the index"
                )
                return
        else:
            logger.warning("Unknown vector store format detected; starting with empty index")
            self.docs = []
            self._doc_vectors = None

        self._rebuild_index()
        logger.info(f"Loaded vector store: {len(self.docs)} docs")

    def save_index(self):
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        payload = {
            "docs": self.docs,
            "vectors": self._doc_vectors,
            "backend": self._index_backend,
        }
        with open(self.vector_store_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Vector store saved")

    def load_pdf(self, pdf_path):
        # import PDF reader lazily and give clear instruction if missing
        try:
            from pypdf import PdfReader
        except Exception:
            try:
                from PyPDF2 import PdfReader
            except Exception:
                raise RuntimeError(
                    "PDF parser not installed. Install with:\n"
                    "  pip install pypdf\n"
                )

        text = ""
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def _batch(self, iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    def _embed_batch(self, texts):
        """
        Embed a list of strings and return numpy array shape (n, dim) dtype float32.
        Uses the detected client and batches requests.
        """
        if not texts:
            # Lazy import numpy only when needed
            try:
                import numpy as np
            except Exception:
                raise RuntimeError("numpy is required for embeddings; install with: pip install numpy")
            return np.zeros((0, 0), dtype="float32")

        if self._embed_type is None or self._embed_client is None:
            if not self._init_sentence_transformer():
                raise RuntimeError("Embedding client not initialized. Install and configure an embedding backend.")

        if self._embed_type == "genai":
            try:
                return self._embed_with_genai(texts)
            except Exception as exc:
                logger.warning(
                    "Gemini embedding failed (%s). Falling back to sentence-transformers.",
                    exc,
                )
                if not self._init_sentence_transformer():
                    raise RuntimeError(
                        "Embedding request to google.genai failed and no local fallback available."
                    ) from exc
                return self._embed_with_sentence_transformer(texts)
        elif self._embed_type == "gga":
            try:
                return self._embed_with_gga(texts)
            except Exception as exc:
                logger.warning(
                    "google.generativeai embedding failed (%s). Falling back to sentence-transformers.",
                    exc,
                )
                if not self._init_sentence_transformer():
                    raise RuntimeError(
                        "Embedding request to google.generativeai failed and no local fallback available."
                    ) from exc
                return self._embed_with_sentence_transformer(texts)
        elif self._embed_type == "sentence-transformer":
            return self._embed_with_sentence_transformer(texts)

        raise RuntimeError("Unsupported embed client type")

    def _ensure_numpy(self):
        try:
            import numpy as np  # noqa: F401
        except Exception:
            raise RuntimeError("numpy is required for embeddings; install with: pip install numpy")

    def _embed_with_genai(self, texts):
        self._ensure_numpy()
        import numpy as np
        client = self._embed_client
        from google.genai import types  # local import to avoid module import errors earlier
        vectors = []
        cfg = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        for batch in self._batch(texts, self.embed_batch_size):
            contents = [
                types.Content(parts=[types.Part(text=text)])
                for text in batch
            ]
            res = client.models.embed_content(
                model=self._embed_model,
                contents=contents,
                config=cfg,
            )
            for emb in res.embeddings:
                vectors.append(np.array(emb.values, dtype="float32"))
        return np.vstack(vectors)

    def _embed_with_gga(self, texts):
        self._ensure_numpy()
        import numpy as np
        gga = self._embed_client
        vectors = []
        for batch in self._batch(texts, self.embed_batch_size):
            for text in batch:
                resp = gga.embeddings.create(model=self._embed_model, input=text)
                vectors.append(np.array(resp.data[0].embedding, dtype="float32"))
        return np.vstack(vectors)

    def _embed_with_sentence_transformer(self, texts):
        self._ensure_numpy()
        import numpy as np
        model = self._embed_client
        try:
            vectors = model.encode(
                texts,
                batch_size=self.embed_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers embedding failed. Ensure the model is installed correctly."
            ) from exc
        return np.asarray(vectors, dtype="float32")

    def embed_texts(self, texts):
        """
        Public wrapper: accepts single string or list of strings.
        """
        contents = texts if isinstance(texts, list) else [texts]
        return self._embed_batch(contents)

    def add_documents(self, file_path):
        # Đọc PDF
        text = self.load_pdf(file_path)
        if not text:
            logger.warning("No text extracted from PDF")
            return

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
        except Exception:
            try:
                from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "LangChain text splitters not available. Install with: pip install 'langchain-text-splitters'"
                ) from exc

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
        if not chunks:
            logger.warning("No chunks produced from document")
            return

        # Tạo vector embedding (batched)
        vectors = self.embed_texts(chunks)
        if vectors.size == 0:
            raise RuntimeError("Embedding produced no vectors")

        # Ensure vectors are C-contiguous float32
        try:
            import numpy as np
        except Exception:
            raise RuntimeError("numpy is required for embeddings; install with: pip install numpy")

        vectors = np.ascontiguousarray(vectors, dtype="float32")

        # Update stored state
        dim = vectors.shape[1]
        if self._doc_vectors is None:
            self._doc_vectors = vectors.copy()
        else:
            import numpy as np

            self._doc_vectors = np.vstack([self._doc_vectors, vectors])

        self.docs.extend(chunks)
        self._rebuild_index(expected_dim=dim)
        self.save_index()
        logger.info(f"Added {len(chunks)} chunks to index")

    def retrieve(self, query, top_k=3):
        if self.index is None or len(self.docs) == 0:
            return []
        q_vecs = self.embed_texts(query)
        if q_vecs.size == 0:
            return []
        try:
            import numpy as np
        except Exception:
            raise RuntimeError("numpy is required for retrieval; install with: pip install numpy")

        q_vec = np.ascontiguousarray(q_vecs[0].reshape(1, -1), dtype="float32")
        top_k = min(top_k, len(self.docs))
        D, I = self.index.search(q_vec, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.docs) and idx >= 0:
                results.append(self.docs[idx])
        return results

    def _create_index(self, dim):
        if self._index_backend == "faiss":
            try:
                import faiss
            except Exception as exc:
                logger.warning("FAISS backend unavailable (%s); switching to numpy fallback", exc)
                self._index_backend = "numpy"
            else:
                return faiss.IndexFlatL2(dim)

        if self._index_backend == "numpy":
            return _NumpyFlatL2Index(dim)

        try:
            import faiss
        except Exception as exc:
            logger.warning("FAISS import failed (%s); using numpy fallback index", exc)
            self._index_backend = "numpy"
            return _NumpyFlatL2Index(dim)

        self._index_backend = "faiss"
        logger.info("Using FAISS for similarity search")
        return faiss.IndexFlatL2(dim)

    def _rebuild_index(self, expected_dim=None):
        if self._doc_vectors is None or self._doc_vectors.size == 0:
            self.index = None
            return

        try:
            import numpy as np
        except Exception:
            raise RuntimeError("numpy is required for embeddings; install with: pip install numpy")

        vectors = np.ascontiguousarray(self._doc_vectors, dtype="float32")
        dim = vectors.shape[1]

        if expected_dim is not None and expected_dim != dim:
            logger.warning(
                "Stored vector dimension (%s) does not match new embeddings (%s); rebuilding from scratch",
                dim,
                expected_dim,
            )

        self.index = self._create_index(dim)
        self.index.add(vectors)
