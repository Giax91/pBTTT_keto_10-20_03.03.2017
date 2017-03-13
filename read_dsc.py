#!/usr/bin/python3

# Modul for reading DSC data into data-objects
import pandas as pd

def readDSC(path):
    '''
    Funktion that reads in DSC .txt files with 'ISO-8859-2' encoding.
    Output is a list of panda DataFrames with attributes (name, date, method and weight )
    '''
    with open(path, 'r', encoding='ISO-8859-2') as f:
        lines = f.readlines()
        date = lines[0].strip().split()[1]
    
        # Extract curve
        count = -1
        capture = False
        strList = []
        curveList = []
        
        for l in lines:
            count += 1
        
            if l == "Curve Name:\n":
                curveName = lines[count+1].strip()
                continue
                
            if l == "                          [°C]           [°C]           [°C]           [mW]\n":
                capture = True
                continue
            
            if l == "Sample:\n":
                weight = lines[count+1].strip().split()[1]
            
            if l == "Method:\n":
                method = lines[count+1].strip()
            
            
            if l == "Results:\n":
                capture = False
                continue
            
            if l == "User:\n":
                curve = pd.DataFrame(strList[:-2],columns=["Index","Abscissa [°C]", "Ts [°C]", "Tr [°C]", "Value [mW]"], dtype="float64")
                curveList.append(curve)
                
                # Add attributes
                curve.name = curveName
                curve.date = date
                curve.method = method
                curve.weight = float(weight)
                strList = []
                continue
            if capture:
                strList.append(l.strip().split())
            
        return curveList
