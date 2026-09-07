# 🚀 VITEXOR

### AI-Powered Code Review & Analysis Platform

> **Intelligent Code Analysis. Better Code.**

VITEXOR is an AI-powered code review platform that analyzes source code and provides structured feedback on potential bugs, security concerns, performance issues, maintainability problems, and possible code improvements.

The system combines **CodeBERT embeddings, FAISS-based semantic retrieval, Gemini LLM analysis, Pydantic validation, and a FastAPI backend** with an interactive web frontend.

---

## ✨ Key Features

- 🤖 AI-powered source code analysis
- 🔍 Semantic code retrieval using CodeBERT
- ⚡ FAISS-based vector similarity search
- 🐛 Bug and runtime issue detection
- 🔐 Security issue analysis
- 🚀 Performance analysis
- 🧹 Maintainability analysis
- 💡 AI-generated code improvement suggestions
- ✅ Structured response validation using Pydantic
- 🌐 FastAPI REST API
- 💻 Interactive web-based code review interface
- 📊 Review metrics and categorized findings
- 📝 Final code quality verdict

---

## 🧠 How VITEXOR Works

VITEXOR follows a retrieval-augmented AI code review pipeline.

```text
                         ┌─────────────────────┐
                         │       User          │
                         │   Source Code       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Frontend       │
                         │ HTML / CSS / JS      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │   REST API Layer    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Review Service    │
                         │ Validation & Flow   │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
              ┌─────────────────┐      ┌─────────────────┐
              │     CodeBERT     │      │      FAISS      │
              │    Embeddings    │─────▶│ Semantic Search │
              └─────────────────┘      └────────┬────────┘
                                                │
                                                ▼
                                     ┌─────────────────────┐
                                     │ Retrieved Context   │
                                     │ + Source Code       │
                                     └──────────┬──────────┘
                                                │
                                                ▼
                                     ┌─────────────────────┐
                                     │     Gemini LLM      │
                                     │    AI Analysis      │
                                     └──────────┬──────────┘
                                                │
                                                ▼
                                     ┌─────────────────────┐
                                     │ Structured Response │
                                     │ Pydantic Validation │
                                     └──────────┬──────────┘
                                                │
                                                ▼
                                     ┌─────────────────────┐
                                     │ Review Results      │
                                     │                     │
                                     │ Issues              │
                                     │ Security            │
                                     │ Performance         │
                                     │ Maintainability     │
                                     │ Improved Code       │
                                     │ Final Verdict       │
                                     └─────────────────────┘
