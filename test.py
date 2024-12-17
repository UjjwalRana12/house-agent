from agents import LLMProcessor
# Create an instance of the LLMProcessor class
llm_processor = LLMProcessor()

# Define your input content (example: some house-related data)
content = """
The house is a 3-bedroom, 2-bathroom property located in a suburban neighborhood. 
It was built in 2005 and has a square footage of 2,200 sqft. 
what must be its price
"""


response = llm_processor.process_with_llm(content)

# Print the response from the model
print(response)
