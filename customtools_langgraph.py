import os, json
import pandas as pd

from langgraph.graph import StateGraph, END
from typing import TypedDict
from pathlib import Path

class GraphState(TypedDict):
    input_path1: str
    input_path2: str
    Patient_Number: int
    df1: list
    df2: list
    merged_df: list

## Node 1
def input_validation_node(state: GraphState):

    path1 = Path(state["input_path1"])
    path2 = Path(state["input_path2"])

    if not (path1.exists() or path2.exists()):
        raise ValueError("File does not exists")
    
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    df1_filtered = df1[df1["Patient_Number"] == state["Patient_Number"]]
    df2_filtered = df2[df2["Patient_Number"] == state["Patient_Number"]]

    return {
        "df1": df1_filtered.to_dict(orient="records"),
        "df2": df2_filtered.to_dict(orient="records")
    }

## Node 2
def feature_engineering_node(state: GraphState):

    df1 = pd.DataFrame(state["df1"])
    df2 = pd.DataFrame(state["df2"])

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

    df2["Physical_activity"] = df2["Physical_activity"].fillna(0)

    return {
        "df1": df1.to_dict(orient="records"),
        "df2": df2.to_dict(orient="records")
    }  

## Node 3:
def merge_node(state: GraphState):

    df1 = pd.DataFrame(state["df1"])  
    df2 = pd.DataFrame(state["df2"])

    df2 = df2.groupby("Patient_Number", as_index=False)["Physical_activity"].mean()
    df2 = df2.rename(columns={"Physical_activity": "Patient_Avg_Steps"})

    merged_df = pd.merge(df1, df2, on="Patient_Number",how="inner")
    merged_df = merged_df[["Patient_Number","Blood_Pressure_Abnormality","Age","BMI","BMI_Risk","Sex","Pregnancy","Smoking","Level_of_Stress","Patient_Avg_Steps"]]

    output_path = r"F:\GEN_AI\Graph_CrewAI\output"
    os.makedirs(output_path, exist_ok=True)

    file_path = f"{output_path}/Patient_num_{state['Patient_Number']}_result.json"
    merged_df.to_json(file_path, orient="records", indent=2)

    return {"merged": merged_df.to_dict(orient="records")}

## Build Graph
builder = StateGraph(GraphState)

builder.add_node("validator",input_validation_node)
builder.add_node("engineering", feature_engineering_node)
builder.add_node("merging",merge_node)

builder.set_entry_point("validator")

builder.add_edge("validator","engineering")
builder.add_edge("engineering","merging")
builder.add_edge("merging",END)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({
        "input_path1": r"F:\GEN_AI\Graph_CrewAI\data\healthcare_dataset1.csv",
        "input_path2": r"F:\GEN_AI\Graph_CrewAI\data\healthcare_dataset2.csv",
        "Patient_Number": int(input("Enter Patient Number: "))        
    })

    print(result)