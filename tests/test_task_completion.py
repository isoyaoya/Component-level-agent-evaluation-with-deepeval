from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate
import asyncio

def run_task_completion_test():
    # Create test case
    test_case_travel_itinerary = LLMTestCase(
        input="Plan a 3-day itinerary for a trip to Paris, including cultural landmarks and local cuisine recommendations.",
        actual_output="Day 1: Visit the Eiffel Tower, have dinner at Le Jules Verne. Day 2: Explore the Louvre Museum, lunch at Angelina Paris. Day 3: Walk through Montmartre, evening at a wine bar.",
        tools_called=[
            ToolCall(
                name="Itinerary Generator",
                description="Generates travel itineraries based on destination and duration.",
                input_parameters={"destination": "Paris", "days": 3},
                output=[
                    "Day 1: Eiffel Tower, Le Jules Verne.",
                    "Day 2: Louvre Museum, Angelina Paris.",
                    "Day 3: Montmartre, wine bar.",
                ],
            ),
            ToolCall(
                name="Restaurant Finder",
                description="Finds top restaurants in a given city.",
                input_parameters={"city": "Paris"},
                output=["Le Jules Verne", "Angelina Paris", "local wine bars"],
            ),
        ],
    )

    # Initialize Bedrock model
    bedrock_model = AmazonBedrockModel(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        region_name="us-east-1"
    )

    task_completion_metric = TaskCompletionMetric(model=bedrock_model)
    
    # Run evaluation
    async def run_eval():
        await evaluate(
            test_cases=[test_case_travel_itinerary],
            metrics=[task_completion_metric],
            display_config=DisplayConfig(verbose_mode=True),
            async_config=AsyncConfig(run_async=True),
        )

    # Run the async function using asyncio
    asyncio.run(run_eval())

if __name__ == "__main__":
    run_task_completion_test()
