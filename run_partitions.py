import papermill as pm
import os

# Create results directory
os.makedirs("results", exist_ok=True)

for partition_num in range(1, 5):
    input_path = f"partition{partition_num}.pkl"
    output_notebook = f"results/Executed_Partition_{partition_num}.ipynb"
    output_result = f"results/results_partition{partition_num}.pkl"
    
    try:
        pm.execute_notebook(
            "features_extraction_partitions.ipynb",
            output_notebook,
            parameters={
                "PARTITION_NUM": partition_num,
                "INPUT_TEMPLATE": input_path,
                "OUTPUT_TEMPLATE": output_result
            }
        )
        print(f"Successfully processed partition {partition_num}")
    except Exception as e:
        print(f"Error processing partition {partition_num}: {str(e)}")