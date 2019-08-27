from flask_restplus import Api, Resource, fields
from flask import Flask, jsonify, request, make_response, abort, render_template, redirect, url_for

app = Flask(__name__)
api = Api(app, version='1.0', title='MuseFind Tagging API',
          description='Automated Tagging By NLP')
ns = api.namespace('MuseFind_api', description='Methods')
single_parser = api.parser()
single_parser.add_argument('n', type=int, required=True, help='first number')
single_parser.add_argument('m', type=int, required=True, help='second number')


def summation(a, b):
    return a+b


@ns.route('/addition')
class Addition(Resource):
    """Uploads your data to the recommender system"""
    @api.doc(parser=single_parser, description='Enter Two Integers')
    def get(self):
        """Uploads a new transaction to Rex (Click to see more)"""
        args = single_parser.parse_args()
        n1 = args.n
        m1 = args.m
        r = summation(n1, m1)
        print(r)
        return {'add': r}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
