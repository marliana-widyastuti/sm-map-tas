import utils
import argparse
import os
import time
import pandas as pd
from datetime import datetime, timedelta

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
# print(vars(args))

# Get current time when script is executed
start_time = time.time()

# retrieve weather data

# data = utils.tasData(date=args.date, output=args.output)
# data.downloadWeather()
# data.uploadToGEE()

##### FOR ANALYSIS _-----------
from datetime import datetime
import pandas as pd

# start date
start_date = datetime.strptime("2022-01-04", "%Y-%m-%d")
end_date = datetime.strptime("2023-07-31", "%Y-%m-%d")

date_list = pd.date_range(start_date, end_date, freq='MS')
date_list = date_list# + pd.DateOffset(days=4)

for day in date_list:
    d = day.strftime('%Y%m%d')

    D = utils.runArea(date=d, res=2000, xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax, model_path=args.modelPath, 
                    output=args.output)
    D.exportImage()
    D.exe_mod(model_path=args.modelPath)
    D.combined_SM()
    D.calc_sm()
    # D.SMtoGCS()

end_time = time.time()
utils.calculate_execution_time(start_time, end_time)