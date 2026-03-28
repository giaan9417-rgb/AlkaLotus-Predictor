# data.py
import pandas as pd

# Dữ liệu 7 hợp chất dựa trên nghiên cứu in silico
ALKALOID_DATA = {
    "Name": ["Nuciferine", "Nornuciferine", "Roemerine", "Pronuciferine", "Liensinine", "Neferine", "Isoliensinine"],
    "Formula": ["C19H21NO2", "C18H19NO2", "C18H17NO2", "C19H21NO3", "C37H42N2O6", "C38H44N2O6", "C37H42N2O6"],
    "MW": [295.4, 281.3, 279.3, 311.4, 610.7, 624.8, 610.7],
    "LogP": [3.3, 3.4, 3.0, 2.5, 5.2, 6.4, 6.4],
    "HBD": [0, 1, 0, 0, 1, 0, 1],
    "HBA": [3, 3, 3, 4, 8, 8, 8],
    "dG_BACE1": [-8.3, -8.3, -9.0, -8.6, -9.6, -9.0, -9.6],
    "dG_AChE": [-8.2, -8.1, -8.6, -8.6, -7.5, -7.5, -7.7],
    "BBB_Permeability": [True, True, True, True, False, False, False], # Giả lập dựa trên logP và MW
    "SMILES": [
        "COC1=C(C2=C(C3=C1)C=CC=C3C4CC(N(CC42)C)C)OC", 
        "COC1=C(C2=C(C3=C1)C=CC=C3C4CC(NCC42))OC", 
        "CN1CCC2=CC3=C(C4=C2C1CC5=CC=CC=C54)OCO3", 
        "COC1=CC2=C(C=C1OC)C3(CCN(C2)C)C=CC(=O)C=C3", 
        "CN1CCC2=CC(=C(C=C2C1CC3=CC=C(C=C3)OC)OC)OC4=C(C=CC(=C4)CC5C6=CC(=C(C=C6CCN5C)OC)O)O", 
        "CN1CCC2=CC(=C(C=C2C1CC3=CC=C(C=C3)OC)OC)OC4=C(C=CC(=C4)CC5C6=CC(=C(C=C6CCN5C)OC)OC)O", 
        "CN1CCC2=CC(=C(C=C2C1CC3=CC=C(C=C3)O)OC)OC4=C(C=CC(=C4)CC5C6=CC(=C(C=C6CCN5C)OC)O)OC"
    ]
}

def get_database():
    return pd.DataFrame(ALKALOID_DATA)