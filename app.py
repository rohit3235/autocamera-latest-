from configmain import *
from imgtests import *
from report import *


app = Flask(__name__)
api = Api(app)


class Places(Resource):
    def post(self):
        # parse request arguments
        parser = reqparse.RequestParser()
        parser.add_argument("camid", required=True)
        parser.add_argument("image1test", required=True)
        parser.add_argument("image2perfect", required=True)
        args = parser.parse_args()
        camid = args.camid
        image1test = args.image1test
        image2perfect = args.image2perfect

        # read test and perfect images
        image1test_path = os.path.join('data', 'TestImages', image1test)
        image2perfect_path = os.path.join('data', 'TestImages', image2perfect)

        # call the testing functions
        test_results = generate_report(
            camid, image1test_path, image2perfect_path)
        test_names = ['CamID',
                      'Image_Not_Blur',
                      'Image_Not_Inverted',
                      'Image_Not_Mirrored',
                      'Image_Not_Rotated',
                      'Image_Horizontal_Shift',
                      'Image_Vertical_Shift',
                      'Image_Not_Cropped_In_ROI',
                      'Image_Has_No_Noise_Staticlines_Scrolling',
                      'SSIM_Score',
                      'Brisque_Score'
                      ]

        # format the results
        dict_results = {test_names[i]: test_results[i]
                        for i in range(0, len(test_names))}
        json_results = json.dumps(
            dict_results, indent=0, sort_keys=False)

        return dict_results, 201


api.add_resource(Places, '/places')

if __name__ == '__main__':
    app.run(debug=True)
