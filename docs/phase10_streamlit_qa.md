# Phase 10: Streamlit Q&A Bug Fix and RAG Presentation Layer Polish

This phase addresses the Streamlit Q&A interaction bug, polishes the FLAN-T5 grounded generation pipeline, fixes visual contrast issues on attention flags, and improves citation and retrieval context inspection.

---

## 1. Q&A State Bug & Root Cause

### Root Cause
In Streamlit's execution model, whenever a widget or button is interacted with, the entire script re-executes top-to-bottom. The previous implementation suffered from several state-management defects:
1. **Transient Question Popping**: `st.session_state.pop("pending_question", "")` attempted to pop question state during widget declaration, causing the text input box to reset to empty on the subsequent rerun cycle when the user submitted a question.
2. **Re-assignment after Instantiation**: Assigning values to a widget's session_state key after widget instantiation created widget key conflicts and caused input reset.
3. **Unprotected Input Reruns**: Asking a question without wrapping the input in `st.form` caused intermediate reruns where state was wiped.

### Fix Implemented
1. **Stable Session State Keys**:
   - `st.session_state["qa_input_text"]`: Governs the current string in the input text box.
   - `st.session_state["qa_history"]`: Stores a list of `(question, QueryResult)` tuples for the active document.
2. **Pre-instantiation Example Question Population**:
   - Clicking an example question button sets `st.session_state["qa_input_text"] = example_text` *before* `st.text_input(key="qa_input_text")` is instantiated, followed by `st.rerun()`. On rerun, Streamlit initializes the text input with the updated key value.
3. **Form Submission Pattern**:
   - Wrapped input inside `st.form(key="qa_form", clear_on_submit=False)` so pressing Enter or clicking "Ask Document" retains the submitted question across reruns.
4. **History & Upload Isolation**:
   - Added a "Clear Q&A History" button to reset `qa_history` without losing document analysis state.
   - Analyzing a newly uploaded PDF automatically resets `qa_history` to prevent cross-document context contamination.

---

## 2. FLAN-T5 Grounded RAG Generation Flow

The architecture follows the modular NLP pipeline:

```
User Question
  └─► MPNet-QA Semantic Clause Retriever (src/retrieval.py)
        └─► RAGContextBuilder (src/rag.py)
              └─► FLAN-T5 Generator (src/generation.py)
                    └─► Grounded Answer + Citations + Evidence Inspector
```

- If retrieval returns an empty context (`context.is_empty`), FLAN-T5 is **never invoked**. The application immediately displays:
  `"No relevant clause context was retrieved for this question."`
- Citations (`Clause ID`, `Title`, `Page`, `Similarity Score`) displayed in the UI are derived **exclusively** from application `RAGContext` objects (`RetrievedClause`).

---

## 3. UI Improvements & Accessibility Fixes

- **Color Contrast Fix**: Resolved text contrast on "Watch Out For" attention flag cards (`#78350f` text on `#fffbeb` background with `#92400e` titles).
- **Document Hero**: Updated header hero banner (`AI-Powered Legal Document Analyzer — Analyze • Understand • Ask`).
- **Retrieval Context Inspector**: Added a dedicated `"View retrieval context"` expandable section allowing users to view the exact clause evidence passed to FLAN-T5.
