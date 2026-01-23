# Minervini SEPA Agentic Trading Workflow - Task Checklist

## Phase 1: Project Setup & Configuration

- [x] **Project Structure Setup**
  - [x] Create project directory structure (`src/`, `config/`, `data/`, `tests/`, `ui/`)
  - [x] Initialize Python virtual environment
  - [x] Create `requirements.txt` with dependencies (yfinance, pandas, streamlit, pydantic, etc.)
  - [x] Set up `.env` file for API keys (Kite Connect credentials for future)

- [x] **Configuration System**
  - [x] Create `config/agent_criteria.yaml` base structure
  - [x] Create `config/trend_template_criteria.yaml`
  - [x] Create `config/fundamental_criteria.yaml`
  - [x] Create `config/vcp_criteria.yaml`
  - [x] Create `config/entry_criteria.yaml`
  - [x] Create `config/risk_criteria.yaml`
  - [x] Create `config/position_criteria.yaml`
  - [x] Create `config/portfolio_criteria.yaml`
  - [x] Implement config loader utility (`src/utils/config_loader.py`)

---

## Phase 2: Data Tools Implementation

- [x] **Stock Universe Tools**
  - [x] Implement `get_all_indian_stocks()` - Fetch NSE + BSE listings
  - [x] Add caching mechanism for stock list (refresh daily)
  - [x] Handle delisted/suspended stocks filtering

- [x] **Market Data Tools**
  - [x] Implement `get_stock_data(symbol, period)` - OHLCV via yfinance
  - [x] Handle `.NS` / `.BO` suffix logic for NSE/BSE
  - [x] Implement rate limiting to avoid API throttling
  - [x] Add data validation and cleaning

- [x] **Technical Analysis Tools**
  - [x] Implement `calculate_moving_averages(symbol)` - 50/150/200 SMAs
  - [x] Calculate SMA trend direction (up/down over N months)
  - [x] Implement `get_relative_strength(symbol, benchmark)` vs NIFTY 500
  - [x] Calculate 52-week high/low percentages

- [x] **Fundamental Data Tools**
  - [x] Implement `get_fundamentals(symbol)` - EPS, revenue, margins
  - [x] Parse quarterly financials from yfinance
  - [x] Calculate YoY growth rates
  - [x] Detect EPS acceleration patterns (Q1 < Q2 < Q3)

- [x] **Pattern Detection Tools**
  - [x] Implement `detect_vcp(symbol, lookback_days)` - VCP pattern detection
  - [x] Identify contraction count and ratios
  - [x] Detect volume dry-up conditions
  - [x] Validate base length (20-65 days)

- [x] **Entry & Execution Tools**
  - [x] Implement `identify_pivot(symbol)` - Find breakout pivot price
  - [x] Implement `calculate_position_size(entry, stop, portfolio_value)`
  - [x] Implement `execute_paper_trade(order)` - Log to JSON
  - [x] Implement `get_portfolio_metrics()` - Win rate, avg gain/loss

---

## Phase 3: Agent Implementation

- [x] **Agent Infrastructure (aisuite)**
  - [x] Configure `aisuite` client (OpenAI/Anthropic support)
  - [x] specific tool definitions for aisuite
  - [x] Implement base Agent class with `aisuite` integration

- [x] **Agent 1: Trend Template Agent**
  - [x] Create `src/agents/trend_template_agent.py`
  - [x] Implement 8-point trend template check:
    - [x] Price above 150-day SMA
    - [x] Price above 200-day SMA
    - [x] 150-day SMA above 200-day SMA
    - [x] 200-day SMA trending up (1+ months)
    - [x] 50-day SMA above 150 and 200
    - [x] Price above 50-day SMA
    - [x] Price 30%+ above 52-week low
    - [x] Price within 25% of 52-week high
  - [x] Implement RS ranking filter (≥70)
  - [x] Output: List of Stage 2 qualifying stocks

- [x] **Agent 2: Fundamental Overlay Agent**
  - [x] Create `src/agents/fundamental_agent.py`
  - [x] Implement EPS growth check (≥20% YoY)
  - [x] Implement EPS acceleration detection
  - [x] Implement revenue growth check (≥15%)
  - [x] Implement margin expansion detection
  - [x] Implement earnings surprise check
  - [x] Output: Quality stocks with strong fundamentals

- [x] **Agent 3: VCP Pattern Agent**
  - [x] Create `src/agents/vcp_pattern_agent.py`
  - [x] Integrate `detect_vcp()` tool
  - [x] Validate contraction criteria from config
  - [x] Score VCP quality (tight patterns score higher)
  - [x] Output: Stocks with valid VCP setups

- [x] **Agent 4: Entry Point Agent**
  - [x] Create `src/agents/entry_point_agent.py`
  - [x] Integrate `identify_pivot()` tool
  - [x] Check volume expansion on breakout (≥1.5x)
  - [x] Verify proximity to 52-week high (≤15%)
  - [x] Prefer first-base patterns
  - [x] Output: Actionable buy signals with pivot price

- [x] **Agent 5: Risk Management Agent**
  - [x] Create `src/agents/risk_management_agent.py`
  - [x] Calculate stop-loss placement (6-7% target, 10% max)
  - [x] Integrate position sizing logic
  - [x] Enforce max 2% portfolio risk per trade
  - [x] Validate no averaging down rule
  - [x] Output: Risk-adjusted trade orders

- [x] **Agent 6: Position Management Agent**
  - [x] Create `src/agents/position_agent.py`
  - [x] Implement free-roll trigger (move stop to breakeven at 2x risk)
  - [x] Implement partial profit taking (at 20% gain)
  - [x] Implement trailing stop using 20-day MA
  - [x] Detect heavy volume breakdown for exit
  - [x] Output: Sell signals and position updates

- [x] **Agent 7: Portfolio Review Agent**
  - [x] Create `src/agents/portfolio_review_agent.py`
  - [x] Track batting average (win rate)
  - [x] Trigger stop tightening below 40% win rate
  - [x] Allow pyramiding only on winners
  - [x] Enforce max 25% single position size
  - [x] Output: Adjusted position sizing recommendations

---

## Phase 4: Pipeline Orchestration

- [x] **Agent Pipeline**
  - [x] Create `src/pipeline/sepa_pipeline.py`
  - [x] Wire agents in sequence: Trend → Fundamental → VCP → Entry → Risk
  - [x] Implement inter-agent data passing
  - [x] Add logging and traceability for each agent step
  - [x] Handle errors gracefully with fallback logic

- [x] **Scheduling**
  - [x] Implement daily scan trigger (market close)
  - [x] Add manual trigger option
  - [x] Store scan results with timestamps

---

## Phase 5: User Interface (Streamlit)

- [x] **Dashboard Layout**
  - [x] Create `ui/streamlit_app.py`
  - [x] Design main dashboard with tabs:
    - [x] Scan Results tab
    - [x] VCP Patterns tab
    - [x] Trade Signals tab
    - [x] Open Positions tab
    - [x] Portfolio Performance tab

- [x] **Agent Trace Visualization**
  - [x] Display agent decision flow
  - [x] Show criteria pass/fail for each stock
  - [x] Visualize VCP patterns with charts

- [x] **Trade Management UI**
  - [x] Human-in-the-loop approval interface
  - [x] Display recommended position size and stop-loss
  - [x] Approve/Reject trade buttons
  - [x] Paper trading log viewer

- [x] **Configuration Editor**
  - [x] UI to modify agent criteria YAML files
  - [x] Preview changes before saving
  - [x] Reset to defaults option

---

## Phase 6: Verification & Testing

- [ ] **Test 1: Trend Template Validation**
  - [ ] Prepare 10 known Stage 2 stocks
  - [ ] Prepare 10 non-qualifying stocks
  - [ ] Run agent and compare to manual analysis
  - [ ] Target: 100% match

- [ ] **Test 2: VCP Pattern Detection**
  - [ ] Collect 5 confirmed historical VCPs
  - [ ] Run detector on historical data
  - [ ] Verify all 5 detected correctly

- [ ] **Test 3: Risk Calculation**
  - [ ] Create 10 scenarios (varying entry/stop/portfolio)
  - [ ] Unit test position sizing calculations
  - [ ] Target: 100% accuracy

- [ ] **Test 4: End-to-End Pipeline**
  - [ ] Run pipeline on NIFTY 50 stocks
  - [ ] Verify complete execution without errors
  - [ ] Validate output signal format

- [ ] **Test 5: Historical Backtest**
  - [ ] Run backtest on 2020-2024 data
  - [ ] Calculate win rate, avg gain/loss
  - [ ] Target: Win rate > 50%, Avg gain > 2x Avg loss

---

## Phase 7: Documentation & Deployment

- [ ] **Documentation**
  - [ ] Create `README.md` with setup instructions
  - [ ] Document each agent's purpose and criteria
  - [ ] Add usage examples and screenshots
  - [ ] Create `CONTRIBUTING.md` for expert modifications

- [ ] **Deployment**
  - [ ] Test Streamlit deployment locally
  - [ ] Configure for Streamlit Cloud (optional)
  - [ ] Set up scheduled execution (cron or workflow)

---

## Pending Expert Input

> [!IMPORTANT]
> The following items require expert criteria confirmation before implementation:

- [ ] **Agent 1 Criteria**: Trend Template specific thresholds
- [ ] **Agent 2 Criteria**: Fundamental overlay requirements
- [ ] **Agent 3 Criteria**: VCP pattern parameters
- [ ] **Agent 4 Criteria**: Entry point specifications
- [ ] **Agent 5 Criteria**: Risk management rules
- [ ] **Agent 6 Criteria**: Position management triggers
- [ ] **Agent 7 Criteria**: Portfolio review thresholds
