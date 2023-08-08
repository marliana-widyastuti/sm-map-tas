import utils
import argparse
import os
import time
from datetime import datetime, timedelta

# if __name__ == '__main__':
parser = argparse.ArgumentParser(
                    prog='run_model.py',
                    description='What the program does',
                    epilog='Text at the bottom of help',
                    usage='run_model.py [-h] [date] [modelPath] [res] [xmin] [xmax] [ymin] [ymax]')

parser.add_argument('--date',help='date of predicted soil moisture [YYYYmmdd]', default='20220104', required=False)
parser.add_argument('--modelPath', default='model/',help='directory path of Existing models [def = model/]', required=False)
parser.add_argument('--output', help='path to save the output files [def = current working directory]', default= os.getcwd(), required=False)
parser.add_argument('--res', help='spatial resolution [meter]', default=80, required=False)
parser.add_argument('--xmin', default= 143.75, help='Xmin [default=143.75]', required=False)
parser.add_argument('--xmax', default= 148.50, help='Xmax [default=148.50]', required=False)
parser.add_argument('--ymin', default= -43.75, help='Ymin [default=-43.75]', required=False)
parser.add_argument('--ymax', default= -39.50, help='Ymax [default=-39.50]', required=False)
args, unknown = parser.parse_known_args()
print(vars(args))

# Get current time when script is executed
start_time = time.time()

# Set start_date, end_date, and forecast_time
start_date = datetime.today() - timedelta(days = 5)
end_date  = start_date + timedelta(days = 1)

date = start_date.strftime('%Y%m%d')
D = utils.runArea(date=date, res=args.res, xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax, model_path=args.modelPath, 
                  output=args.output)
D.exportImage()
D.exe_mod(model_path=args.modelPath)
D.combined_SM()
D.calc_sm()

to_day = datetime.today().strftime('%Y%m%d')

data = utils.tasData(to_day, output=args.output)
data.downloadWeather()
data.uploadToGEE()

end_time = time.time()
utils.calculate_execution_time(start_time, end_time)