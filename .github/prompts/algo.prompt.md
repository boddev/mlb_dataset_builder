Create a Python prediction algorithm using Logistic Regression with Gradient Boosting to forecast player statistics based on historical performance data. The model should:

Input Requirements:
- Accept a target date parameter for testing data selection
- Process data from three CSV files:
  * 2024_batting.csv (batting statistics)
  * 2024_pitching.csv (pitching statistics)
  * 2024_linescore.csv (game results)

Output Predictions:
1. Batter Statistics (for each player in lineup):
   - Last 10 games overall performance metrics
   - Last 10 home games performance metrics
   - Last 10 away games performance metrics

2. Starting Pitcher Statistics:
   - Last 10 games overall performance metrics
   - Last 10 home games performance metrics
   - Last 10 away games performance metrics

3. Team Performance Metrics:
   - Overall win-loss record
   - Home game win-loss record
   - Away game win-loss record

Technical Requirements:
- Implement Logistic Regression with Gradient Boosting
- Include data validation and error handling
- Optimize for computational efficiency
- Provide confidence intervals for predictions
- Handle missing or incomplete data scenarios

Documentation Requirements:
- Include input data format specifications
- Document feature engineering process
- Provide model evaluation metrics
- Add usage examples with sample data

Please ensure the implementation follows machine learning best practices and includes appropriate model validation techniques.