from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import os
import ast
import json
####
from report import *
from imgtests import *
from configmain import *
####

app = Flask(__name__)
api = Api(app)

@app.route('/')
def index():
    return app.make_response("hello world world")


class Places(Resource):
    def post(self):
    # parse request arguments
        parser = reqparse.RequestParser()
        parser.add_argument("camid", required=True)
        parser.add_argument("image1test", required=True)
        parser.add_argument("image2perfect", required=True)
        args = parser.parse_args()
##############
        camid =args.camid
        image1test_path = os.path.join(IMAGE_FOLD_PATH,'data/TestImages', args.image1test)
        image2perfect_path = os.path.join(IMAGE_FOLD_PATH,'data/TestImages',args.image2perfect)

        test_results = generate_report(camid, image1test_path, image2perfect_path)
        #test_names = ['CamId','Blur','check_scale','noise','scrolled','allign','mirror','blackspots','ssim_score','brisque_score']
        test_names = ['CamId','Blur','check_scale','noise','scrolled','allign','mirror','blackspots','ssim_score','staticlines','rotation_deg']
        #print("pass:0/fail:1or>1")
        dict_results = {test_names[i]: test_results[i] for i in range(0,len(test_names))}
        json_results = json.dumps(dict_results, indent = 4)  

        return json_results,201


api.add_resource(Places, '/places')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
