<span>This is a personal project for the AI for Business subject.</span>
<br>
Taking the research paper titled "Dynamic Pricing Strategy Driven by Deep Reinforcement Learning with Empirical Analysis on the Collaborative Optimization of Hotel Revenue Management and Customer Satisfaction" as the core foundation of the development process, combined with insights from other related studies, and considering the actual local market conditions, I propose a Demand-based Dynamic Pricing Model with the following structure:
<img width="2480" height="925" alt="image" src="https://github.com/user-attachments/assets/b251d7de-06a3-43a3-b71f-0cd3d4a3e62f" />
- Multi-datasource Encoding: Performs preprocessing of data from multiple sources, enabling the model to capture both internal and external fluctuations affecting the hotel.
- RPT (Room Demand Prediction Transformer): Forecasts room demand for the next day.
- PPO (Proximal Policy Optimization): Based on the predicted demand and historical data, generates the optimal pricing strategy for the following day.
