import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai import Agent, Task, Crew, Process
from typing import Type
import os, json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

### Input File Validation
class InputValidationSchema(BaseModel):
    input_path1: str = Field(..., description="input path for file 1")
    input_path2: str = Field(..., description="input path for file 2")
    Patient_Number: int = Field(..., description="Patient id from user input")
    model_config = {"arbitrary_types_allowed": True}

class InputValidationTool(BaseTool):
    name: str = "input_validation_tool"
    description: str = "Validation of input files."
    args_schema: Type[BaseModel] = InputValidationSchema

    def _run(self, input_path1: str, input_path2: str, Patient_Number: int) -> pd.DataFrame:
        
        path1 = Path(input_path1)
        path2 = Path(input_path2)

        if not (path1.exists() or path2.exists()):
            return json.dumps({
                "status": "Error",
                "message": "File does not exists, Please check!!"
            })
            
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)

        df1_filtered = df1[df1["Patient_Number"] == Patient_Number]
        df2_filtered = df2[df2["Patient_Number"] == Patient_Number]

        return {"df1_filtered": df1_filtered.head(10).to_dict(orient="records"), "df2_filtered": df2_filtered.head(10).to_dict(orient="records")}

## Feature Engineering
class FeatureEngineeringSchema(BaseModel):
    df1: list #pd.DataFrame = Field(..., description="First DataFrame")
    df2: list #pd.DataFrame = Field(..., description="Second DataFrame")
    model_config = {"arbitrary_types_allowed": True}

class FeatureEngineeringTool(BaseTool):
    name: str = "feature_engineering_tool"
    description: str = "to clean the data from both dataframes"
    args_schema: Type[BaseModel] = FeatureEngineeringSchema

    def _run(self, df1: list, df2: list) -> pd.DataFrame:

        df1 = pd.DataFrame(df1)
        df2 = pd.DataFrame(df2)

        df1["Blood_Pressure_Abnormality"] = df1["Blood_Pressure_Abnormality"].map(lambda x: "Normal" if x == 0 else "Abnormal")
        df1["Sex"] = df1["Sex"].map(lambda x: "Male" if x == 0 else "Female")
        df1["Pregnancy"] = df1["Pregnancy"].map({0: "No" , 1: "Yes"}).fillna("NA")
        df1["Smoking"] = df1["Smoking"].map(lambda x: "No" if x == 0 else "Yes")
        df1["Level_of_Stress"] = df1["Level_of_Stress"].map({1: "Low", 2: "Normal", 3: "High"})
        df1["BMI_Risk"] = df1["BMI"].apply(lambda x: 
                                           "Underweight - Nutritional deficiency, weak immunity" if x < 18.5 else 
                                           "Normal weight - Low risk" if x > 18.5 and x < 24.9 else
                                           "Overweight - Overweight risk" if x > 25 and x < 29.9 else
                                           "Obesity Class I - High risk" if  x > 30 and x < 34.9 else
                                           "Obesity Class II - Very high risk" if  x > 35 and x < 39.9 else
                                           "Obesity Class III (Severe) - Extremely high risk"
                                           )

        df2['Physical_activity'] = df2["Physical_activity"].fillna(0)

        return {"df1":df1.head(10).to_dict(orient="records"), "df2" : df2.head(10).to_dict(orient="records")}

## Analytics Engine for Merging DataFrames
class AnalyticsEngineSchema(BaseModel):
    df1: list # pd.DataFrame = Field(..., description="First DataFrame")
    df2: list # pd.DataFrame = Field(..., description="Second DataFrame")
    Patient_Number: int = Field(..., description="Patient id from user input")

    model_config = {
        "arbitrary_types_allowed": True
    }

class AnalyticsEngineTool(BaseTool):
    name: str = "analytics_engine_tool"
    description: str = "to merge the dataframes"
    args_schema: Type[BaseModel] = AnalyticsEngineSchema

    def _run(self, df1: list, df2: list, Patient_Number: int) -> pd.DataFrame:

        df1 = pd.DataFrame(df1)
        df2 = pd.DataFrame(df2)

        print(df1.columns)
        print(df2.columns)

        if "Patient_Number" not in df1.columns or "Patient_Number" not in df2.columns:
            return json.dumps({
                "status": "Error",
                "message": "Key Column not found"
            })
        
        #df1_filtered = df1[df1["Patient_Number"] == Patient_Number]
        #df2_filtered = df2[df2["Patient_Number"] == Patient_Number]

        #print(df1.head(1))
        #print(df2.head(5))

        df2 = df2.groupby("Patient_Number",as_index=False)["Physical_activity"].mean().rename(columns={"Physical_activity": "Patient_Avg_Steps"})
        #df2_filtered["Patient_Avg_Steps"] = df2_filtered["Patient_Number"].map(avg_steps)

        merged_df = pd.merge(df1, df2, on="Patient_Number", how="inner")

        merged_df = merged_df[["Patient_Number","Blood_Pressure_Abnormality","Age","BMI","BMI_Risk","Sex","Pregnancy","Smoking","Level_of_Stress","Patient_Avg_Steps"]]


        output_path = r"F:\GEN_AI\Graph_CrewAI\output"
        os.makedirs(output_path, exist_ok=True)

        output_file = f"{output_path}/Patient_num_{Patient_Number}_result.json"
        merged_df.to_json(output_file, orient='records', indent=2, index=False)
        
        return merged_df.head(10).to_dict(orient="records")
        #return f"Execution Completed and Output file has been save to {output_file}"

input_validation_tool = InputValidationTool()
feature_engineering_tool = FeatureEngineeringTool()
analytics_engine_tool = AnalyticsEngineTool()

validator_agent = Agent(
    role="Input Validator Assistant",
    goal="Need to Validate the input paths and return Pandas Dataframes",
    backstory="You are expert in validating the input paths and have knowledge on pandas library",
    tools=[input_validation_tool],
    verbose=True
)

engineering_agent = Agent(
    role="Feature Engineering Assistant",
    goal="Need to do feature engineering on provided data",
    backstory="You are expert Python pandas library",
    tools=[feature_engineering_tool],
    verbose=True
)

merging_agent = Agent(
    role="Data Merge Assistant",
    goal="Need to merge the pandas dataframes and return single merged dataframe using patient id {Patient_Number}",
    backstory="You are expert in validating the input paths and have knowledge on pandas library",
    tools=[analytics_engine_tool],
    verbose=True
)

validator_task = Task(
    description= "Validate the input paths given {input_path1} and {input_path2} and filter using {Patient_Number}",
    agent=validator_agent,
    expected_output="return the pandas dataframes"
)

engineering_task = Task(
    description= "You need do some feature engineering using pandas library",
    agent=engineering_agent,
    expected_output="return the pandas dataframes",
    context=[validator_task]
)

merging_task = Task(
    description= "Merged the both the dataframes from previous task using patient id {Patient_Number}",
    agent=merging_agent,
    expected_output="return the merged pandas dataframe and save it to 'F:\GEN_AI\Graph_CrewAI\output' path",
    context=[engineering_task],
    max_retries=5
)

crew = Crew(
    agents=[validator_agent, engineering_agent, merging_agent],
    tasks=[validator_task, engineering_task, merging_task],
    process= Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff(inputs={
        "input_path1": "F:\GEN_AI\Graph_CrewAI\data\healthcare_dataset1.csv",
        "input_path2": "F:\GEN_AI\Graph_CrewAI\data\healthcare_dataset2.csv",
        "Patient_Number": input("Enter Patient_Number here: ")
    })

    print(result)


"""
df1 = pd.read_csv("F:\GEN_AI\Graph_CrewAI\data\healthcare_dataset1.csv")
df2 = pd.read_csv("F:\GEN_AI\Graph_CrewAI\data\healthcare_dataset2.csv") #['Patient_Number', 'Day_Number', 'Physical_activity']

print(df1[['Blood_Pressure_Abnormality','Sex','Pregnancy','Smoking','Level_of_Stress']].head(5))
print(df2[['Patient_Number', 'Day_Number', 'Physical_activity']].head(5))
print(merged_df[["Patient_Number","Sex","Physical_activity"]].head(5))
merged_df = pd.merge(df1, df2, on="Patient_Number", how="inner")
"""

