## Third-party code / models / data (VYASA-1)

This project uses free/open-source libraries and downloads free datasets/models at runtime.

### Python libraries
- **FastAPI** (web server)
- **Uvicorn** (ASGI server)
- **ChromaDB** (vector database)
- **sentence-transformers** (embeddings + cross-encoder reranker)
- **transformers** (Hugging Face pipelines)
- **indic-transliteration** (Devanagari → IAST transliteration)

### Hugging Face models (downloaded automatically)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Emotion**: `j-hartmann/emotion-english-distilroberta-base`

### Datasets (downloaded automatically)
- `bhavykhatri/DharmicData` (GitHub download in our scripts)
- `atmabodha/Vedanta_Datasets` (GitHub download in our scripts)
- `rahular/itihasa` (GitHub download in our scripts)
- GRETIL Sanskrit corpus zip: `https://gretil.sub.uni-goettingen.de/gretil/1_sanskr.zip`

### Note
Licenses and terms belong to the respective authors. If you plan to publish this system, review each dataset/model license and attribution requirements.

