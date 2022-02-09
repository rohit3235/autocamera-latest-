# -*- coding: utf-8 -*-
r"""
[Summary]
This is the old main.py file, the start file
This file is used to test the functions by running this file in command prompt by typing the
following command 
main.py, camid, test_img_path,perfect_img_path

use relative path
python main.py --camid 1 --image1test autocameratest2\data\TestImages\bluecolortint.png --image2perfect autocameratest2\data\TestImages\perfect.png

:param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
:type [ParamName]: [ParamType](, optional)
...
:raises [ErrorType]: [ErrorDescription]
...
:return: [ReturnDescription]
:rtype: [ReturnType]
"""
from report import *
from imgtests import *
from configmain import *
import argparse

#def main()-> None:
    #start_setup()
parser = argparse.ArgumentParser()

parser.add_argument('--camid', type=str, required = True)
parser.add_argument('--image1test', type=str, required=True)
parser.add_argument('--image2perfect', type=str, required=True)
args = parser.parse_args()
####
camid =args.camid
#image1test_path = os.path.join(ROOT_DIR,args.image1test)
#image2perfect_path = os.path.join(ROOT_DIR,args.image2perfect)
import pathlib

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ''))
#in windows use \ backlash and in linux using / below
image1test_path = pathlib.Path.cwd().joinpath('data', 'TestImages', args.image1test)
image2perfect_path = pathlib.Path.cwd().joinpath('data', 'TestImages', args.image2perfect)
#image1test_path = os.path.join(ROOT_DIR,"data/TestImages", args.image1test)
#image2perfect_path = os.path.join(ROOT_DIR,"data/TestImages",args.image2perfect)


#image1test_path = os.path.join(os.path.dirname(__file__),IMAGE_FOLD_PATH,'\data\TestImages', args.image1test)
#image2perfect_path = os.path.join(os.path.dirname(__file__),'\data\TestImages',args.image2perfect)



#test_results = generate_report(args.camid, args.image1test, args.image2perfect)
test_results = generate_report(camid,image1test_path,image2perfect_path)
#test_names = ['CamId','Blur','check_scale','noise','scrolled','allign','mirror','blackspots','ssim_score','brisque_score']
test_names = ['CamId','Blur','check_scale','noise','scrolled','allign','mirror','blackspots','ssim_score','staticlines','rotation_deg']
print("pass:0/fail:1or>1")
for i in range(0,len(test_names)):
    print(f"{test_names[i]}: {test_results[i]}")    
 

