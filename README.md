# Install dependencies
pip install openai anthropic google-generativeai

# Run benchmark
python main.py

File Structure

├── env.py              # Game engine with bank trades, robber
├── agents.py           # LLM and random agents
├── orchestrator.py     # Game loop with hallucination tracking
├── metrics.py          # Enhanced metrics calculation
├── main.py             # Main benchmark script
├── llm_clients.py      # API wrappers (no changes)
└── secrets.py          # API keys (gitignored)

Sample Output
🏆 Win Rates
OpenAI_gpt-5-nano        : 35%
Claude_Haiku_4.5         : 40%
Gemini_2.5_Flash         : 20%
Random_baseline          : 5%

🧠 Hallucination Stats
Claude_Haiku_4.5:
  Total decisions: 245
  Index errors: 3
  Action failures: 12
  Hallucination rate: 6.1%
  Penalty score: 0.695

💰 Trade Behavior
OpenAI_gpt-5-nano:
  Player trades: 5
  Bank trades: 23
  Total trades: 28
  
⚡ Resource Efficiency
Claude_Haiku_4.5:
  Build rate: 0.187 builds/turn
  Avg final resources: 8.3
  Efficiency score: 0.102


💡 NEXT STEPS:
   1. ✅ Hallucination tracking is now accurate
   2. ✅ Bank trades are working
   3. ✅ Robber mechanic implemented
   4. 🔜 Add port trades (3:1, 2:1)
   5. 🔜 Build proper hex board
   6. 🔜 Dynamic scenarios (resource scarcity)
   7. 🔜 Strategy pivot detection
