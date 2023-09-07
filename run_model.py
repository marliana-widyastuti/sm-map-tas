import utils
import argparse
import os
import time
import pandas as pd
from datetime import datetime, timedelta
import sys

to_day = datetime.today().strftime('%Y%m%d')

parser = argparse.ArgumentParser(
                    prog='run_model.py',
                    description='What the program does',
                    epilog='Text at the bottom of help',
                    usage='run_model.py [-h] [date] [modelPath] [res] [xmin] [xmax] [ymin] [ymax]')

parser.add_argument('--date',help='date of predicted soil moisture [YYYYmmdd]', default=to_day, required=False)
parser.add_argument('--modelPath', default='model/',help='directory path of Existing models [def = model/]', required=False)
parser.add_argument('--output', help='path to save the output files [def = current working directory]', default= os.getcwd(), required=False)
parser.add_argument('--res', help='spatial resolution [meter]', default=80, required=False)
parser.add_argument('--xmin', default= 143.50, help='Xmin [default=143.75]', required=False)
parser.add_argument('--xmax', default= 149.00, help='Xmax [default=148.50]', required=False)
parser.add_argument('--ymin', default= -44.00, help='Ymin [default=-43.75]', required=False)
parser.add_argument('--ymax', default= -39.00, help='Ymax [default=-39.50]', required=False)
args, unknown = parser.parse_known_args()
print(vars(args))

filename  = open(f"log/printed-{args.date}.log",'w')
sys.stdout = filename
print("Anything printed will go to the output file")

# Get current time when script is executed
start_time = time.time()

# retrieve weather data #comment this section if you run earlier than 4 days ago
data = utils.tasData(date=to_day, output=args.output)
data.downloadWeather()
data.uploadToGEE()

#run the model
D = utils.runArea(date=args.date, res=args.res, xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax, model_path=args.modelPath, 
                  output=args.output)
D.exportImage()
D.exe_mod()
D.combined_SM()
D.calc_sm()
D.SMtoGCS()

end_time = time.time()
utils.calculate_execution_time(start_time, end_time)