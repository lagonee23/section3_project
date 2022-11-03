from flask import Flask

def create_app():
    app = Flask(__name__)

    from routes import routes
    # from yourapplication.views.frontend import frontend
    app.register_blueprint(routes.coin)
    # app.register_blueprint(frontend)

    return app

if __name__ == "__main__":
  app = create_app()
  app.run(debug=True)