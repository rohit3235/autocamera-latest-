# app_bkp02Feb
from configmain import *
from imgtests import *
####
from report import *

####

app = Flask(__name__)
api = Api(app)

# @app.route('/')
# def hello_world():
#    return "Hello Docker"


class Places(Resource):
    def post(self):
        # parse request arguments
        parser = reqparse.RequestParser()
        parser.add_argument("camid", required=True)
        parser.add_argument("image1test", required=True)
        parser.add_argument("image2perfect", required=True)
        args = parser.parse_args()
##############
        camid = args.camid
        image1test = args.image1test
        image2perfect = args.image2perfect
        # image1test_path = os.path.join(IMAGE_FOLD_PATH,'\autocameratest2\data\TestImages',args.image1test)
        # image2perfect_path = os.path.join(IMAGE_FOLD_PATH,'\autocameratest2\data\TestImages',args.image2perfect)

        image1test_path = os.path.join('data', 'TestImages', image1test)
        image2perfect_path = os.path.join('data', 'TestImages', image2perfect)

        # image1test_path = pathlib.Path.cwd().joinpath('data', 'TestImages', args.image1test)
        # image2perfect_path = pathlib.Path.cwd().joinpath('data', 'TestImages', args.image2perfect)
        # image1test_path = pathlib.Path('data', 'TestImages', args.image1test)
        # image2perfect_path = pathlib.Path('data', 'TestImages', args.image2perfect)

        test_results = generate_report(
            camid, image1test_path, image2perfect_path)
        # test_names = ['CamId','Blur','check_scale','noise','scrolled','allign','mirror','blackspots','ssim_score','brisque_score']
        # test_names = ['CamId', 'Blur', 'check_scale', 'noise', 'scrolled', 'align', 'shift',
        #               'mirror', 'blackspots', 'ssim_score', 'staticlines', 'rotation_deg']
        # print("pass:0/fail:1or>1")
        # test_names = ['CamID',
        #               'not_inverted',
        #               'not_mirrored',
        #               'rotation',
        #               'not_cropped_in_ROI_region',
        #               'no_noise_staticline_scrolling',
        #               'blur',
        #               'check_scale',
        #               'noise',
        #               'scrolled',
        #               'rgb_layer_align',
        #               'image_shift',
        #               'mirror',
        #               'blackspots',
        #               'static_lines',
        #               'ssim_score',
        #               'brisque_score']
        test_names = ['CamID',
                      'Image_Not_Inverted',
                      'Image_Not_Mirrored',
                      'Image_Not_Rotated',
                      'Image_Horizontal_Shift',
                      'Image_Vertical_Shift',
                      'Image_Not_Cropped_In_ROI',
                      'Image_Has_No_Noise_Staticlines_Scrolling',
                      'SSIM',
                      'Brisque_Score'
                      ]
        dict_results = {test_names[i]: test_results[i]
                        for i in range(0, len(test_names))}
        json_results = json.dumps(
            dict_results, indent=4, sort_keys=False)

        return json_results, 201


api.add_resource(Places, '/places')

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    # app.run(host='0.0.0.0', port= 8080, debug=True)
    app.run(debug=True)
