# Antutor Metric AI Backend

A high-performance, Python-based FastAPI backend for a multi-agent learning support platform (Sejong University capstone project). The system utilizes a novel hybrid evaluation pipeline—an initial **NLI-based Gatekeeper** for contradiction filtering, followed by **LLM Atomic Decomposition** and **Multi-Agent Feedback**—to provide intelligent scoring, personalized scaffolding, and educational insights to learners.

---

## 🏗️ Core Architecture

The platform operates on a robust, asynchronous pipeline designed for cost efficiency and low latency:

1. **NLI Contradiction Gatekeeper**: Quickly filters out answers heavily contradicting the ground truth using a local DeBERTa Cross-Encoder. This acts as an early exit layer before invoking expensive LLMs.
2. **The Antutor Metric (Atomic NLI)**: Uses the Gemini API to decompose a user's answer into atomic propositions and evaluates them against the ground truth using NLI Entailment to dynamically score knowledge alignment.
3. **Parallel RAG Retrieval**: Concurrently fetches real-time external data (News API and Knowledge Graphs) to be injected into the expert context.
4. **Multi-Agent Expert System**: Three expert personas evaluate the answer in parallel using Azure **Phi-4**:
   - 🎓 *The Academic Auditor*: Evaluates theoretical accuracy and formal principles.
   - 📈 *The Market Practitioner*: Evaluates real-world market impacts based on recent news.
   - 🌐 *The Macro-Connector*: Evaluates broader macroeconomic trends using knowledge graphs.
5. **Dynamic Scaffolding Moderator**: Identifies the weakest domain (lowest expert score) and synthesizes an encouraging hint using Gemini to guide the learner without giving away the direct answer.

---

## 📁 File Structure

- `main.py`: The core FastAPI application containing APIs, AI model initializations, and the asynchronous orchestration logic for the multi-agent system.
- `config.py`: Centralized configuration variables including Target Educational Concepts, Concept Dictionary, Agent Prompts, and environment keys.
- `requirements.txt`: Python package dependencies (FastAPI, sentence-transformers, bcrypt, PyJWT, Google GenAI, etc.).
- `.env` *(Not included in repo)*: Environment variables (API Keys, Endpoints).

---

## 🔧 Detailed Function Descriptions (`main.py`)

### 🔐 Authentication & Authorization
- `get_password_hash` / `verify_password`: Utility functions implementing `bcrypt` for secure password hashing.
- `create_access_token`: Generates short-lived JSON Web Tokens (JWT) for secure session management.
- `get_current_user`: FastAPI dependency that validates the JWT structure, checks database existence, and authorizes requests.
- `register` / `login_for_access_token`: Onboarding and token-generation endpoints.

### 🧠 AI Processing & Evaluation
- `extract_atomic_propositions`: Invokes the Gemini 3 Flash model to break an entire paragraph into discrete, independent atomic knowledge units.
- `verify_nli_contradiction`: The *fast-pass gatekeeper*. Invokes a local DeBERTa V3 model to check if the user's answer is an explicit contradiction (Label 0).
- `verify_nli_atomic`: Calculates the final Antutor Score by determining what percentage of the extracted propositions map to an *Entailment* label (Label 1) against the ground truth.
- `calculate_antutor_score`: An asynchronous wrapper that sequentially executes the extraction and NLI entailment logic as a single task block.

### 🔌 External Context Retrieval (RAG)
- `retrieve_news_rag`: Asynchronously fetches recent articles from the News API based on the concept to provide real-world context for the Market Practitioner.
- `retrieve_knowledge_graph`: Currently a simulation wrapper that demonstrates fetching logical connected nodes to support the Macro-Connector.

### 🤖 Multi-Agent Logic & Orchestration
- `call_expert_agent`: A generic HTTP wrapper handling requests to the Azure Phi-4 endpoint. It injects specific prompt templates and contextual data (NLI score, News, or KG) depending on the persona requested.
- `generate_moderator_guidance_message`: Consumes the lowest scoring expert's feedback and uses Gemini to synthesize a friendly, pedagogical nudge directly addressed to the student.

### 💬 Chat & Session Flow
- `start_session`: Initializes session memory, tracks scaffold levels/attempts, and provides the first conceptual question.
- `chat`: The orchestration HTTP endpoint. It coordinates give-up keyword detection, early exit contradiction filtering, parallel Atomic-NLI scoring, parallel context retrieval, and parallel multi-agent evaluation to deliver a low-latency, personalized response.
- `end_session`: Concludes the active learning session, processes radar charts for growth visualizations, and calculates final bonus scores based on learner independence.
