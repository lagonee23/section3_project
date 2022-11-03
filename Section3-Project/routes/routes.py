from flask import Blueprint

coin = Blueprint('coin', __name__, url_prefix='/coin')

@coin.route('/')
def index():
    return 'Coin index page'