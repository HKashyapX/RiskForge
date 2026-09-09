# RiskForge

RiskForge is an **AI/NLP Safety Intelligence Platform** built to automatically detect Serious Injury & Fatality (SIF) precursors in unstructured industrial safety reports, unsafe-act observations, and near-miss logs.

## The Pipeline & Flow

The RiskForge platform bridges the gap between raw, messy human-written safety logs and structured, actionable HSE (Health, Safety & Environment) metrics.

### 1. Data Ingestion & NLP Pipeline (Backend)
- **Raw Input:** The system accepts unstructured safety narratives (e.g., "A worker fell from a 10ft ladder but sustained no injuries").
- **NLP Processing:** A Python-based AI pipeline processes these logs to:
  - Extract the core **Hazard Class** (e.g., `fall_from_height`, `fire_explosion`).
  - Calculate the **SIF Potential Score** (predicting how likely the event was to cause a serious fatality).
  - Identify failed **Safety Barriers** and involved **Assets/Locations**.
- **Routing:** Based on the SIF Potential, the AI assigns a routing flag:
  - `critical_escalation`: High SIF potential, requires immediate management attention.
  - `hitl_review`: Ambiguous logs requiring a Human-In-The-Loop review.
  - `auto_dismiss`: Low-risk observations.

### 2. Frontend Intelligence Engine
The web interface (React/Vite) dynamically ingests this processed data to provide real-time intelligence to safety officers.

#### Overview Dashboard
The command center. It computes the **SIF Precursor Density (SPD)**, a critical metric showing the percentage of total reports that contained a hidden fatality risk. It provides a 12-week rolling trend graph, top hazard distributions, and recent critical escalations.

#### Incident Logs
A robust, searchable queue of all safety reports. Safety officers can filter logs by their AI-assigned status (Escalated, Pending Review, etc.). Clicking a log reveals the AI's detailed breakdown of the narrative.

#### Escalations
A dedicated inbox specifically for logs flagged as `critical_escalation`. This guarantees that high-energy events (like a dropped 2-ton load that missed a worker by inches) never get buried under low-risk reports (like a tripped hazard).

### Assets & Safety Barriers
Two analytical views that flip the perspective:
- **Assets:** Shows which physical locations or equipment (e.g., forklifts, compressors) are generating the most SIF precursors.
- **Barriers:** Highlights which critical safety controls (e.g., Fall Protection, Lockout/Tagout) are failing most frequently across the organization.

#### Pattern Analysis (Analytics)
A comprehensive breakdown of IOGP Life-Saving Rule violations, high-risk activities, and emerging hazards.

## 🛠 Tech Stack
- **Frontend:** React, TypeScript, TailwindCSS, Vite
- **Data Layer:** Python, Pandas, JSONL (for scalable AI data processing)
- **Deployment:** Static Build (`dist/`)

## Running Locally

1. Install dependencies:
   ```bash
   npm install
   ```
2. Run the development server:
   ```bash
   npm run dev
   ```
3. Build for production:
   ```bash
   npm run build
   ```
   *(Note: The `dist/` directory is the final compiled artifact ready for static hosting).*
