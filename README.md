# QUANTUM TUMOR SENSING MACHINE

> **Tagline:** Intelligent Brain MRI Analysis & Tumor Classification Platform  
> **Status:** Production-Ready Full-Stack Academic & Clinical Decision Support System  
> **Model Accuracy:** 96.24% Benchmark Accuracy (EfficientNetB3 Functional Architecture)  

---

## 🔬 System Overview

**QUANTUM TUMOR SENSING MACHINE** is a full-stack medical AI platform engineered for deep learning-driven brain magnetic resonance imaging (MRI) analysis, multi-class tumor classification, explainable AI visualization via mathematical Grad-CAM, integrated clinical consultation powered by Groq LLaMA 3.3 70B, and quantum machine learning (QML) research simulations.

### Key Capabilities

1. **Pre-Trained Deep Learning Inference Engine:**
   - Directly integrates the user's pre-trained functional Keras model (`brain_tumor_model_final trained 96% .keras`).
   - Backbone: **EfficientNetB3** with 385 sublayers, internal Rescaling and Normalization layers.
   - Input Tensor: `(None, 300, 300, 3)` RGB float32 in $[0, 255]$.
   - Output Layer: `dense_2` with 3 softmax units producing calibrated posterior probabilities for:
     - **Glioma Tumor**
     - **Meningioma Tumor**
     - **Pituitary Tumor**
   - Verified benchmark evaluation: **96.24% test accuracy**, **0.9618 macro F1-score**.

2. **True Mathematical Grad-CAM Explainability:**
   - Computes exact gradients of the top predicted class score with respect to the `top_activation` convolutional feature maps (`(10, 10, 1536)`) using TensorFlow's `tf.GradientTape`.
   - Global average pooling across spatial feature maps yields importance weights.
   - Generates dual explainability outputs:
     - **Pure Colorized Heatmap:** Jet colormap visualization of isolated neural activation foci.
     - **Alpha-Blended Overlay:** 60% MRI scan + 40% Grad-CAM heatmap for anatomical localization.
   - Interactive UI toggle between **Overlay**, **Pure Heatmap**, **Side-by-Side Comparison**, and **Raw MRI**.

3. **Quantum Computing Research Module:**
   - Dedicated theoretical and simulated Quantum Machine Learning (QML) section.
   - Formulates Variational Quantum Classifiers (VQC) and Quantum Kernel Estimation (QKE):
     - **4-Qubit VQC:** Angle encoding ($R_x, R_y$ on Bloch sphere) + linear CNOT entanglement (16-D Hilbert space).
     - **8-Qubit VQC:** Higher-order Pauli feature mapping ($U_{\Phi}(\mathbf{x})$) + circular all-to-all entanglement (256-D Hilbert space).
     - **16-Qubit QKE:** Amplitude encoding with non-linear ZZ-feature mapping ($65,536$-D Hilbert space) for ambiguous border-zone histological slices.
   - Clear architectural distinction between classical CNN inference and quantum simulations.

4. **Context-Aware AI Medical Assistant:**
   - Powered by Groq Cloud API (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`).
   - Ingests active MRI inference telemetry (predicted subtype, confidence percentage, probability distribution) directly into clinical system prompt.
   - Dual interface:
     - **Docked Right Panel:** Embedded directly in `/analyze` workspace for simultaneous radiologic review.
     - **Standalone Assistant (`/assistant`):** Dedicated chat interface with conversation history and clinical presets.
   - Embedded offline clinical engine fallback for zero downtime when an external API key is unconfigured.

5. **Publication-Grade Clinical PDF Reports:**
   - Automated report generation using ReportLab (`QT-REP-XXXXX`).
   - Includes patient metadata, original MRI scan, Grad-CAM heatmap, probability distribution table, technical inference metadata, clinical diagnostic discussion, and regulatory disclaimers.

---

## 🎨 UI/UX Design & Architecture

The platform features a modern, clinical **Liquid-Glass Design System**:
- **Looping Video Scrim**: High-definition neural cranial video background with dynamic ambient scrim for contrast and depth.
- **Translucent Glassmorphism (`.liquid-glass`)**: Frosted blur, subtle gradient borders via CSS `mask-composite: exclude`, inset highlights, and interactive hover glow states (`.liquid-glass-interactive`). Zero solid black void boxes.
- **Typography & Aesthetics**: Inter font with tight letter-tracking, minimalist custom SVG emblems, and floating liquid-glass navigation pills.
- **Top-Right Profile Hub**: Instant access to:
  - **Operator Profile** (`/profile`): Verified practitioner credentials (#RAD-8824), neuro-radiology permissions, EfficientNetB3 core allocation, and encrypted session telemetry.
  - **Prediction History** (`/history`): Telemetry case archive, liquid-glass filter pills (Glioma, Meningioma, Pituitary), search bar, and Grad-CAM spatial activation modal.
  - **Clinical Reports** (`/reports`): Translucent report table, formal PDF downloads, confidence ratings, and clinical disclaimers.
  - **Tele-Radiology Consult** (`/consult`): Second opinion portal with standard 24h & urgent (&lt;2h) escalation tiers.
  - **System Help** (`/help`): Step-by-step scanning guide, Grad-CAM hotspot color interpretation, model specifications, and AI assistant prompt guide.
  - **Settings** (`/settings`): Custom Groq API key configuration for LLaMA 3.3, practitioner profile updater, and system runtime specifications.

---

## 🔑 Pre-Seeded Default Accounts

For rapid testing and evaluation, the following pre-configured test credentials are initialized on startup:

| Role | Email Address | Password | Permissions |
|---|---|---|---|
| **Administrator** | `admin@quantumtumor.ai` | `AdminPass123!` | System settings, all scans, patient records, admin logs |
| **Quantum Researcher** | `researcher@quantumtumor.ai` | `ResearchPass123!` | Model performance, quantum circuits, inference telemetry |
| **Medical Professional** | `doctor@quantumtumor.ai` | `DoctorPass123!` | MRI analysis, patient directory, clinical AI assistant, PDF reports |

---

## 🚀 Quickstart & Installation

### Option 1: Native Local Setup

#### Prerequisites
- Python 3.10+ (Python 3.12 verified)
- Node.js 18+ (Node.js 20+ recommended)
- Virtualenv

#### 1. Backend Setup
```bash
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI backend server (auto-creates SQLite quantum_tumor.db and seeds accounts)
python run.py
```
Backend API will be running at: `http://localhost:8000`  
Swagger API Docs available at: `http://localhost:8000/docs`

#### 2. Frontend Setup
```bash
cd frontend

# Install Node dependencies
npm install

# Start Next.js development server
npm run dev
```
Frontend Web Application will be live at: `http://localhost:3000`

---

### Option 2: Docker Compose

```bash
# Build and run backend, frontend, and PostgreSQL containers
docker-compose up --build
```
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- PostgreSQL: `localhost:5432` (database: `quantum_tumor_db`)

---

## 🧪 Verification & Automated Testing

### Backend Test Suite (Pytest)
```bash
cd backend
KERAS_HOME="../.keras_home" MPLCONFIGDIR="../.matplotlib_home" PYTHONPATH=. venv/bin/python -m pytest tests/ -v
```
**Verification Scope:**
- Model loading and architecture parameter integrity.
- Real inference accuracy and probability summation ($\sum p_i \approx 1.0$).
- Mathematical Grad-CAM heatmap and overlay generation on disk.
- User authentication, JWT issuance, and role enforcement.
- Sample MRI analysis workflow end-to-end.
- Publication PDF report generation.
- Chatbot context injection and offline fallback engine.
- Quantum research specification API endpoint (`/api/performance/metrics`).
- Patient CRUD operations.

### Frontend Production Build (Next.js)
```bash
cd frontend
npm run build
```
**Verification Scope:**
- TypeScript strict static type checking.
- Next.js Partial Prerendering and static page optimization across all 18 routes.
- Tailwind CSS styling and responsive layout verification.

---

## 🛡️ Medical Regulatory Disclaimer

> **IMPORTANT:** QUANTUM TUMOR SENSING MACHINE is an advanced academic, educational, and investigational decision-support research platform. It is not approved by regulatory bodies (such as the US FDA or EMA) as a standalone primary diagnostic medical device. Model classifications, Grad-CAM heatmaps, and AI Assistant recommendations must be reviewed and correlated by a qualified neuroradiologist and confirmed by histological tissue biopsy prior to initiating clinical interventions.
