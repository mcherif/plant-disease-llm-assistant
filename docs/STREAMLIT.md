# Streamlit UI

A minimal web UI to classify an image and ask a RAG-backed LLM for guidance.

Features
- Image classifier (ViT finetuned) shows top predictions.
- Auto-detects plant and disease from top-1 label and pre-fills the query.
- RAG answers grounded in the KB; uses plant/disease to focus retrieval.
- Default question “What can I do to treat this?” when the box is empty.

Prerequisites
- Models and index:
  - models/vit-finetuned/ (classifier)
  - models/index/kb-faiss-bge/ (FAISS index)
- OPENAI_API_KEY set in your environment.

Run
- With Make (Windows):
  make ui
- Or directly:
  set PYTHONPATH=.&& streamlit run src\interface\streamlit_app.py

Settings (sidebar)
- Index dir: models/index/kb-faiss-bge
- Top-k context: number of chunks to pass to the LLM.
- Device: cpu or cuda.

Workflow
1) Upload an image (jpg/png). Top predictions will show.
2) The UI stores detected_plant and detected_disease in session state.
3) Check “Use detected plant/disease (if available)” to auto-fill inputs.
4) Press “Get answer”.
   - If the text box is empty, it uses: “What can I do to treat this?”
   - The query is enriched with plant/disease and passed to the RAG pipeline.

Troubleshooting
- ModuleNotFoundError: No module named 'src'
  - Run with PYTHONPATH set: set PYTHONPATH=.&& streamlit run src\interface\streamlit_app.py
- Streamlit session_state modification error
  - The app initializes session_state before creating text_input widgets.
- Off-topic answers
  - Ensure plant/disease are populated. Increase Top-k and verify sources are peach-specific, etc.